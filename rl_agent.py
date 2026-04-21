"""
rl_agent.py
===========
DQN tabanlı Takviyeli Öğrenme ajanı.
SUMO/TraCI ortamında dinamik rota yeniden hesaplama yapar.

Mimari:
  State  → 3 boyutlu vektör (konum, hedef mesafesi, önündeki yoğunluk)
  Action → 4 ayrık eylem  (ileri, sağa, sola, bekle)
  Reward → +100 hedefe varış / -10 trafik cezası / -1 her adım

Kullanım:
    python rl_agent.py --train     # Eğitim modu
    python rl_agent.py --test      # Test modu (eğitilmiş ağırlıkları yükler)
"""

import argparse
import random
import os
import pickle
from collections import deque

import numpy as np

# Derin öğrenme: önce tensorflow dene, yoksa numpy tabanlı Q-table kullan
try:
    import tensorflow as tf
    from tensorflow import keras
    DL_AVAILABLE = True
    print("[RL] TensorFlow bulundu — DQN modu aktif.")
except ImportError:
    DL_AVAILABLE = False
    print("[RL] TensorFlow bulunamadı — Q-Table (tabular) moduna geçildi.")

import traci

# -------------------------------------------------------------------
# YAPILANDIRMA
# -------------------------------------------------------------------
NET_FILE        = "osm.net.xml.gz"
WEIGHTS_FILE    = "dqn_weights.weights.h5"
QTABLE_FILE     = "q_table.pkl"

STATE_SIZE      = 6     # [x, y, hedef_x, hedef_y, yoğunluk_ileri, yoğunluk_sağ/sol]
ACTION_SIZE     = 4     # ileri=0, sağa=1, sola=2, bekle=3

EPISODES        = 300
MAX_STEPS       = 500
BATCH_SIZE      = 64
MEMORY_SIZE     = 10_000

GAMMA           = 0.95      # Gelecek ödül indirim faktörü
EPSILON_START   = 1.0       # Keşif oranı başlangıç
EPSILON_END     = 0.05      # Minimum keşif
EPSILON_DECAY   = 0.995     # Her bölüm sonunda düş
LEARNING_RATE   = 0.001

REWARD_GOAL_REACHED   =  100.0
REWARD_TRAFFIC_PENALTY = -10.0
REWARD_STEP_COST       =  -1.0
REWARD_WRONG_TURN      =  -5.0
TRAFFIC_THRESHOLD       =  8     # Kaç araç varsa "tıkalı" kabul edilir


# ===================================================================
# 1. ORTAM SARMALAYICI  (SUMO ↔ Gym benzeri arayüz)
# ===================================================================
class SUMOEnv:
    """
    SUMO simülasyonunu standart Gym benzeri bir arayüze sarar.
    Her episode'da araç sıfırlanır, hedef rastgele seçilir.
    """

    def __init__(self, net_file: str, initial_route: list[str] | None = None):
        import sumolib
        self.net = sumolib.net.readNet(net_file)
        self.edges = [
            e for e in self.net.getEdges()
            # DÜZELTME 1: RL aracımız sadece araç yollarını (passenger) seçsin!
            if not e.getID().startswith(":") and e.allows("passenger")
        ]
        self.initial_route = initial_route   # GA'dan gelen rota
        self.agent_id      = "delivery_vehicle"
        self.current_edge  = None
        self.goal_edge     = None
        self.step_count    = 0

    # ------------------------------------------------------------------
    def reset(self) -> np.ndarray:
        """Yeni bölüm başlatır. Araç konumunu ve hedefi sıfırlar."""
        self.step_count = 0

        # Eski aracı (eğer hala yoldaysa) kaldırmayı dene
        try:
            if self.agent_id in traci.vehicle.getIDList():
                traci.vehicle.remove(self.agent_id)
        except:
            pass

        # ÇÖZÜM: Her bölüm için yepyeni bir ARAÇ ID'si ve ROTA ID'si üretiyoruz!
        self.agent_id = f"rl_car_{random.randint(100000, 999999)}"
        unique_route_id = f"agent_route_{random.randint(100000, 999999)}"

        # Başlangıç konumu: GA rotasının ilk noktası veya rastgele
        if self.initial_route:
            start_edge = self.initial_route[0]
            self.goal_edge = self.initial_route[-1]
        else:
            start_edge     = random.choice(self.edges).getID()
            self.goal_edge = random.choice(self.edges).getID()
            while self.goal_edge == start_edge:
                self.goal_edge = random.choice(self.edges).getID()

        self.current_edge = start_edge

        # Aracı simülasyona ekle
        try:
            traci.route.add(unique_route_id, [start_edge])
            traci.vehicle.add(self.agent_id, unique_route_id, typeID="car")
            traci.vehicle.changeTarget(self.agent_id, self.goal_edge)
            traci.simulationStep()
        except traci.exceptions.TraCIException as e:
            print(f"[Env] Reset hatası: {e}")

        return self._get_state()

    # ------------------------------------------------------------------
    def _get_state(self) -> np.ndarray:
        """
        State vektörü: [norm_x, norm_y, norm_gx, norm_gy, yoğunluk_1, yoğunluk_2]
        """
        try:
            if self.agent_id not in traci.vehicle.getIDList():
                return np.zeros(STATE_SIZE, dtype=np.float32)

            # Anlık konum
            x, y   = traci.vehicle.getPosition(self.agent_id)
            # Hedef kenarın merkezi
            goal_e = self.net.getEdge(self.goal_edge)
            gx     = (goal_e.getFromNode().getCoord()[0] + goal_e.getToNode().getCoord()[0]) / 2
            gy     = (goal_e.getFromNode().getCoord()[1] + goal_e.getToNode().getCoord()[1]) / 2

            # Normalizasyon sınırları (Kadıköy haritası için yaklaşık)
            X_MAX, Y_MAX = 5000.0, 5000.0

            # Araç önündeki 2 kenardaki yoğunluk
            current_edge = traci.vehicle.getRoadID(self.agent_id)
            density1     = min(traci.edge.getLastStepVehicleNumber(current_edge), 20) / 20.0
            # Komşu kenar
            edges_ahead  = traci.vehicle.getNextTLS(self.agent_id)
            density2     = 0.0
            if edges_ahead:
                next_edge_id = current_edge  # Basitleştirilmiş
                density2 = min(traci.edge.getLastStepVehicleNumber(next_edge_id), 20) / 20.0

            return np.array([
                x / X_MAX,
                y / Y_MAX,
                gx / X_MAX,
                gy / Y_MAX,
                density1,
                density2
            ], dtype=np.float32)

        except traci.exceptions.TraCIException:
            return np.zeros(STATE_SIZE, dtype=np.float32)

    # ------------------------------------------------------------------
    def step(self, action: int) -> tuple[np.ndarray, float, bool]:
        """
        Eylemi uygular, simülasyonu bir adım ilerletir.
        Returns: (next_state, reward, done)
        """
        self.step_count += 1
        reward = REWARD_STEP_COST
        done   = False

        try:
            if self.agent_id not in traci.vehicle.getIDList():
                return np.zeros(STATE_SIZE, dtype=np.float32), reward, True

            current_edge = traci.vehicle.getRoadID(self.agent_id)
            density      = traci.edge.getLastStepVehicleNumber(current_edge)

            # Trafik cezası
            if density > TRAFFIC_THRESHOLD:
                reward += REWARD_TRAFFIC_PENALTY

            # Eylem: RL'nin kararına göre rota değiştir
            if action in (1, 2):  # Sağa veya sola
                # Alternatif rotaya yönlendir
                alt_goal = random.choice(self.edges).getID()
                try:
                    traci.vehicle.rerouteTraveltime(self.agent_id)
                except Exception:
                    pass

            elif action == 3:  # Bekle
                traci.vehicle.setSpeed(self.agent_id, 0)
                reward += -2  # Ekstra bekleme cezası
            else:             # İleri — hızı normale döndür
                traci.vehicle.setSpeed(self.agent_id, -1)  # -1 = SUMO'nun kendi hızı

            traci.simulationStep()
            next_state = self._get_state()

            # Hedefe varış kontrolü
            current_edge = traci.vehicle.getRoadID(self.agent_id) if self.agent_id in traci.vehicle.getIDList() else ""
            if current_edge == self.goal_edge or self.agent_id not in traci.vehicle.getIDList():
                reward += REWARD_GOAL_REACHED
                done = True

            # Maksimum adım
            if self.step_count >= MAX_STEPS:
                done = True

        except traci.exceptions.TraCIException as e:
            next_state = np.zeros(STATE_SIZE, dtype=np.float32)
            done = True

        return next_state, reward, done


# ===================================================================
# 2. DQN MODELİ  (TensorFlow varsa)
# ===================================================================
def build_dqn_model(state_size: int, action_size: int):
    """
    İki gizli katmanlı tam bağlı sinir ağı.
    Çıktı: Her eylem için Q-değeri.
    """
    model = keras.Sequential([
        keras.layers.Dense(64, activation="relu", input_shape=(state_size,)),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(action_size, activation="linear")
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE), loss="mse")
    return model


class DQNAgent:
    """Deep Q-Network ajanı (Experience Replay + Target Network)."""

    def __init__(self, state_size: int = STATE_SIZE, action_size: int = ACTION_SIZE):
        self.state_size  = state_size
        self.action_size = action_size
        self.epsilon     = EPSILON_START

        self.memory         = deque(maxlen=MEMORY_SIZE)
        self.model          = build_dqn_model(state_size, action_size)
        self.target_model   = build_dqn_model(state_size, action_size)
        self.update_target()

        self._train_step = 0

    def update_target(self):
        """Hedef ağı güncelle (periyodik)."""
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state: np.ndarray) -> int:
        """ε-greedy politikası ile eylem seç."""
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        q_values = self.model.predict(state[np.newaxis], verbose=0)
        return int(np.argmax(q_values[0]))

    def replay(self):
        """Bellekten mini-batch örnekle ve sinir ağını güncelle."""
        if len(self.memory) < BATCH_SIZE:
            return

        batch = random.sample(self.memory, BATCH_SIZE)
        states      = np.array([t[0] for t in batch])
        actions     = np.array([t[1] for t in batch])
        rewards     = np.array([t[2] for t in batch])
        next_states = np.array([t[3] for t in batch])
        dones       = np.array([t[4] for t in batch])

        # Bellman denklemi ile hedef Q hesapla
        q_current = self.model.predict(states, verbose=0)
        q_next    = self.target_model.predict(next_states, verbose=0)

        for i in range(BATCH_SIZE):
            target = rewards[i]
            if not dones[i]:
                target += GAMMA * np.max(q_next[i])
            q_current[i][actions[i]] = target

        self.model.fit(states, q_current, epochs=1, verbose=0)
        self._train_step += 1

        # Her 10 eğitimde hedef ağı güncelle
        if self._train_step % 10 == 0:
            self.update_target()

        # Epsilon azalt
        if self.epsilon > EPSILON_END:
            self.epsilon *= EPSILON_DECAY

    def save(self, path: str = WEIGHTS_FILE):
        self.model.save_weights(path)
        print(f"[DQN] Ağırlıklar kaydedildi → {path}")

    def load(self, path: str = WEIGHTS_FILE):
        if os.path.exists(path):
            self.model.load_weights(path)
            self.update_target()
            self.epsilon = EPSILON_END  # Test modunda keşif yok
            print(f"[DQN] Ağırlıklar yüklendi ← {path}")
        else:
            print(f"[DQN] Ağırlık dosyası bulunamadı: {path}")


# ===================================================================
# 3. Q-TABLE (TensorFlow yoksa)
# ===================================================================
class QTableAgent:
    """
    Basit tabular Q-Learning.
    State, ayrık bölmelere (bin) dönüştürülür.
    """

    def __init__(self, state_size: int = STATE_SIZE, action_size: int = ACTION_SIZE, bins: int = 10):
        self.action_size = action_size
        self.bins        = bins
        self.epsilon     = EPSILON_START
        self.q_table: dict = {}

    def _discretize(self, state: np.ndarray) -> tuple:
        return tuple((state * self.bins).astype(int))

    def act(self, state: np.ndarray) -> int:
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        key = self._discretize(state)
        if key not in self.q_table:
            return random.randrange(self.action_size)
        return int(np.argmax(self.q_table[key]))

    def learn(self, state, action, reward, next_state, done):
        key      = self._discretize(state)
        next_key = self._discretize(next_state)

        if key not in self.q_table:
            self.q_table[key] = np.zeros(self.action_size)
        if next_key not in self.q_table:
            self.q_table[next_key] = np.zeros(self.action_size)

        td_target = reward + (0 if done else GAMMA * np.max(self.q_table[next_key]))
        self.q_table[key][action] += LEARNING_RATE * (td_target - self.q_table[key][action])

        if self.epsilon > EPSILON_END:
            self.epsilon *= EPSILON_DECAY

    def save(self, path: str = QTABLE_FILE):
        with open(path, "wb") as f:
            pickle.dump(self.q_table, f)
        print(f"[Q-Table] Kaydedildi → {path}")

    def load(self, path: str = QTABLE_FILE):
        if os.path.exists(path):
            with open(path, "rb") as f:
                self.q_table = pickle.load(f)
            self.epsilon = EPSILON_END
            print(f"[Q-Table] Yüklendi ← {path}")


# ===================================================================
# 4. EĞİTİM DÖNGÜSÜ
# ===================================================================
def train(initial_route: list[str] | None = None):
    """
    Ana eğitim döngüsü.
    initial_route: GA'dan gelen edge ID listesi (opsiyonel)
    """
    from background_traffic import BackgroundTrafficManager, generate_sumo_config, SUMO_CFG_FILE, SUMO_BINARY

    # DÜZELTME 2: İf bloğunu sildik. Her eğitimde trafiği baştan, hatasız olarak kursun.
    manager = BackgroundTrafficManager(NET_FILE)
    manager.write_route_file(num_vehicles=100)
    generate_sumo_config(NET_FILE, "background.rou.xml")

    env   = SUMOEnv(NET_FILE, initial_route)
    agent = DQNAgent() if DL_AVAILABLE else QTableAgent()

    # --ignore-route-errors ekleyerek yol bulunamayan araçların simülasyonu çökertmesini engelliyoruz
    traci.start([SUMO_BINARY, "-c", SUMO_CFG_FILE, "--no-warnings", "--no-step-log", "--ignore-route-errors"])
    print("[Eğitim] SUMO başlatıldı.")

    episode_rewards = []

    for episode in range(EPISODES):
        state   = env.reset()
        total_r = 0.0

        for _ in range(MAX_STEPS):
            action              = agent.act(state)
            next_state, reward, done = env.step(action)
            total_r            += reward

            if DL_AVAILABLE:
                agent.remember(state, action, reward, next_state, done)
                agent.replay()
            else:
                agent.learn(state, action, reward, next_state, done)

            state = next_state
            if done:
                break

        episode_rewards.append(total_r)
        avg = np.mean(episode_rewards[-20:]) if len(episode_rewards) >= 20 else total_r

        if episode % 10 == 0:
            print(f"  Bölüm {episode:3d}/{EPISODES} | "
                  f"Ödül: {total_r:7.1f} | "
                  f"Ort(20): {avg:7.1f} | "
                  f"ε: {agent.epsilon:.3f}")

    traci.close()

    # Kaydet
    if DL_AVAILABLE:
        agent.save(WEIGHTS_FILE)
    else:
        agent.save(QTABLE_FILE)

    print("\n[Eğitim] Tamamlandı!")
    return agent


# ===================================================================
# 5. TEST MODU
# ===================================================================
def test(initial_route: list[str] | None = None):
    """Eğitilmiş ajanı tek bir bölümde çalıştırır ve metrikleri döndürür."""
    from background_traffic import SUMO_CFG_FILE, SUMO_BINARY

    env   = SUMOEnv(NET_FILE, initial_route)
    agent = DQNAgent() if DL_AVAILABLE else QTableAgent()

    weight_path = WEIGHTS_FILE if DL_AVAILABLE else QTABLE_FILE
    agent.load(weight_path)

    # --ignore-route-errors ekleyerek yol bulunamayan araçların simülasyonu çökertmesini engelliyoruz
    traci.start([SUMO_BINARY, "-c", SUMO_CFG_FILE, "--no-warnings", "--no-step-log", "--ignore-route-errors"])
    state     = env.reset()
    total_r   = 0.0
    steps     = 0
    done      = False

    while not done and steps < MAX_STEPS:
        action              = agent.act(state)
        state, reward, done = env.step(action)
        total_r            += reward
        steps              += 1

    traci.close()
    print(f"\n[Test] Adım: {steps} | Toplam Ödül: {total_r:.1f}")
    return {"steps": steps, "total_reward": total_r}


# ===================================================================
# CLI
# ===================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RL Ajanı — Dinamik Rota")
    parser.add_argument("--train", action="store_true", help="Eğitim modunu başlat")
    parser.add_argument("--test",  action="store_true", help="Test modunu başlat")
    args = parser.parse_args()

    if args.train:
        train()
    elif args.test:
        test()
    else:
        print("Kullanım: python rl_agent.py --train  veya  --test")