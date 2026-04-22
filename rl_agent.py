"""
rl_agent.py  (v3 — Final)
=========================
Değişiklikler v2 → v3:
  - STATE_SIZE 6 → 8: ilerideki 3 kenarın yoğunluğu da state'e eklendi
  - Dur-kalk CO2 cezası eklendi (speed < 1.0 m/s → -2 puan)
  - EPISODES 300 → 3000, EPSILON_DECAY 0.995 → 0.9985
  - MEMORY_SIZE 10k → 50k
  - DQN gizli katmanlar 64 → 128 nöron
  - En iyi ağırlıklar ayrıca _best.h5 olarak kaydediliyor
  - NET_FILE: osm_cleaned.net.xml
"""

import argparse
import random
import os
import pickle
from collections import deque

import numpy as np

try:
    import tensorflow as tf
    from tensorflow import keras
    DL_AVAILABLE = True
    print("[RL] TensorFlow bulundu — DQN modu aktif.")
except ImportError:
    DL_AVAILABLE = False
    print("[RL] TensorFlow bulunamadı — Q-Table moduna geçildi.")

import traci

NET_FILE     = "osm_cleaned.net.xml"
WEIGHTS_FILE = "dqn_weights.weights.h5"
QTABLE_FILE  = "q_table.pkl"

STATE_SIZE  = 8    # [x, y, gx, gy, d0, d1, d2, d3]
ACTION_SIZE = 4    # ileri, saga, sola, bekle

EPISODES     = 3000
MAX_STEPS    = 1000
BATCH_SIZE   = 64
MEMORY_SIZE  = 50_000

GAMMA         = 0.95
EPSILON_START = 1.0
EPSILON_END   = 0.05
EPSILON_DECAY = 0.9985
LEARNING_RATE = 0.001

REWARD_GOAL_REACHED    =  100.0
REWARD_TRAFFIC_PENALTY =  -10.0
REWARD_STEP_COST       =   -1.0
REWARD_SLOW_PENALTY    =   -2.0
TRAFFIC_THRESHOLD      =   8


class SUMOEnv:
    def __init__(self, net_file: str, initial_route: list[str] | None = None):
        import sumolib
        self.net = sumolib.net.readNet(net_file)
        self.edges = [
            e for e in self.net.getEdges()
            if not e.getID().startswith(":") and e.allows("passenger")
        ]
        self.initial_route = initial_route
        self.agent_id   = "delivery_vehicle"
        self.goal_edge  = None
        self.step_count = 0

    def reset(self) -> np.ndarray:
        self.step_count = 0
        try:
            if self.agent_id in traci.vehicle.getIDList():
                traci.vehicle.remove(self.agent_id)
        except Exception:
            pass

        self.agent_id   = f"rl_car_{random.randint(100000, 999999)}"
        unique_route_id = f"agent_route_{random.randint(100000, 999999)}"

        if self.initial_route:
            start_edge     = self.initial_route[0]
            self.goal_edge = self.initial_route[-1]
        else:
            start_edge     = random.choice(self.edges).getID()
            self.goal_edge = random.choice(self.edges).getID()
            while self.goal_edge == start_edge:
                self.goal_edge = random.choice(self.edges).getID()

        try:
            traci.route.add(unique_route_id, [start_edge])
            traci.vehicle.add(self.agent_id, unique_route_id, typeID="car")
            traci.vehicle.changeTarget(self.agent_id, self.goal_edge)
            traci.simulationStep()
        except traci.exceptions.TraCIException as e:
            print(f"[Env] Reset hatasi: {e}")

        return self._get_state()

    def _get_state(self) -> np.ndarray:
        try:
            if self.agent_id not in traci.vehicle.getIDList():
                return np.zeros(STATE_SIZE, dtype=np.float32)

            x, y   = traci.vehicle.getPosition(self.agent_id)
            goal_e = self.net.getEdge(self.goal_edge)
            gx = (goal_e.getFromNode().getCoord()[0] + goal_e.getToNode().getCoord()[0]) / 2
            gy = (goal_e.getFromNode().getCoord()[1] + goal_e.getToNode().getCoord()[1]) / 2
            X_MAX = Y_MAX = 5000.0

            route_edges = traci.vehicle.getRoute(self.agent_id)
            route_idx   = traci.vehicle.getRouteIndex(self.agent_id)

            def ahead_density(offset: int) -> float:
                idx = route_idx + offset
                if 0 <= idx < len(route_edges):
                    return min(traci.edge.getLastStepVehicleNumber(route_edges[idx]), 20) / 20.0
                return 0.0

            return np.array([
                x / X_MAX,
                y / Y_MAX,
                gx / X_MAX,
                gy / Y_MAX,
                ahead_density(0),
                ahead_density(1),
                ahead_density(2),
                ahead_density(3),
            ], dtype=np.float32)

        except traci.exceptions.TraCIException:
            return np.zeros(STATE_SIZE, dtype=np.float32)

    def step(self, action: int) -> tuple[np.ndarray, float, bool]:
        self.step_count += 1
        reward = REWARD_STEP_COST
        done   = False

        try:
            if self.agent_id not in traci.vehicle.getIDList():
                return np.zeros(STATE_SIZE, dtype=np.float32), reward, True

            current_edge  = traci.vehicle.getRoadID(self.agent_id)
            density       = traci.edge.getLastStepVehicleNumber(current_edge)
            current_speed = traci.vehicle.getSpeed(self.agent_id)

            if density > TRAFFIC_THRESHOLD:
                reward += REWARD_TRAFFIC_PENALTY

            if current_speed < 1.0:
                reward += REWARD_SLOW_PENALTY

            if action in (1, 2):
                try:
                    traci.vehicle.rerouteTraveltime(self.agent_id)
                except Exception:
                    pass
            elif action == 3:
                traci.vehicle.setSpeed(self.agent_id, 0)
            else:
                traci.vehicle.setSpeed(self.agent_id, -1)

            traci.simulationStep()
            next_state = self._get_state()

            current_edge = traci.vehicle.getRoadID(self.agent_id) \
                if self.agent_id in traci.vehicle.getIDList() else ""
            if current_edge == self.goal_edge or \
               self.agent_id not in traci.vehicle.getIDList():
                reward += REWARD_GOAL_REACHED
                done = True

            if self.step_count >= MAX_STEPS:
                done = True

        except traci.exceptions.TraCIException:
            next_state = np.zeros(STATE_SIZE, dtype=np.float32)
            done = True

        return next_state, reward, done


def build_dqn_model(state_size: int, action_size: int):
    model = keras.Sequential([
        keras.layers.Input(shape=(state_size,)),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dense(64,  activation="relu"),
        keras.layers.Dense(action_size, activation="linear"),
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="mse"
    )
    return model


class DQNAgent:
    def __init__(self, state_size: int = STATE_SIZE, action_size: int = ACTION_SIZE):
        self.state_size  = state_size
        self.action_size = action_size
        self.epsilon     = EPSILON_START
        self.memory       = deque(maxlen=MEMORY_SIZE)
        self.model        = build_dqn_model(state_size, action_size)
        self.target_model = build_dqn_model(state_size, action_size)
        self.update_target()
        self._train_step = 0

    def update_target(self):
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, s, a, r, s2, done):
        self.memory.append((s, a, r, s2, done))

    def act(self, state: np.ndarray) -> int:
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        q = self.model.predict(state[np.newaxis], verbose=0)
        return int(np.argmax(q[0]))

    def replay(self):
        if len(self.memory) < BATCH_SIZE:
            return
        batch       = random.sample(self.memory, BATCH_SIZE)
        states      = np.array([t[0] for t in batch])
        actions     = np.array([t[1] for t in batch])
        rewards     = np.array([t[2] for t in batch])
        next_states = np.array([t[3] for t in batch])
        dones       = np.array([t[4] for t in batch])

        q_cur  = self.model.predict(states,      verbose=0)
        q_next = self.target_model.predict(next_states, verbose=0)

        for i in range(BATCH_SIZE):
            target = rewards[i] + (0 if dones[i] else GAMMA * np.max(q_next[i]))
            q_cur[i][actions[i]] = target

        self.model.fit(states, q_cur, epochs=1, verbose=0)
        self._train_step += 1

        if self._train_step % 10 == 0:
            self.update_target()

    def save(self, path: str = WEIGHTS_FILE):
        self.model.save_weights(path)
        print(f"[DQN] Kaydedildi -> {path}")

    def load(self, path: str = WEIGHTS_FILE):
        if os.path.exists(path):
            self.model.load_weights(path)
            self.update_target()
            self.epsilon = EPSILON_END
            print(f"[DQN] Yuklendi <- {path}")
        else:
            print(f"[DQN] Agirlik bulunamadi: {path}")


class QTableAgent:
    def __init__(self, state_size=STATE_SIZE, action_size=ACTION_SIZE, bins=10):
        self.action_size = action_size
        self.bins    = bins
        self.epsilon = EPSILON_START
        self.q_table: dict = {}

    def _d(self, state): return tuple((state * self.bins).astype(int))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        k = self._d(state)
        return int(np.argmax(self.q_table.get(k, np.zeros(self.action_size))))

    def learn(self, s, a, r, s2, done):
        k, k2 = self._d(s), self._d(s2)
        if k  not in self.q_table: self.q_table[k]  = np.zeros(self.action_size)
        if k2 not in self.q_table: self.q_table[k2] = np.zeros(self.action_size)
        td = r + (0 if done else GAMMA * np.max(self.q_table[k2]))
        self.q_table[k][a] += LEARNING_RATE * (td - self.q_table[k][a])
        if self.epsilon > EPSILON_END:
            self.epsilon *= EPSILON_DECAY

    def save(self, path=QTABLE_FILE):
        with open(path, "wb") as f: pickle.dump(self.q_table, f)
        print(f"[Q-Table] Kaydedildi -> {path}")

    def load(self, path=QTABLE_FILE):
        if os.path.exists(path):
            with open(path, "rb") as f: self.q_table = pickle.load(f)
            self.epsilon = EPSILON_END
            print(f"[Q-Table] Yuklendi <- {path}")


def train(initial_route: list[str] | None = None):
    from background_traffic import (
        BackgroundTrafficManager, generate_sumo_config,
        SUMO_CFG_FILE, SUMO_BINARY
    )
    manager = BackgroundTrafficManager(NET_FILE)
    manager.write_route_file(num_vehicles=100)
    generate_sumo_config(NET_FILE, "background.rou.xml")

    env   = SUMOEnv(NET_FILE, initial_route)
    agent = DQNAgent() if DL_AVAILABLE else QTableAgent()
    # EKLENECEK SATIR: 1 saatlik emeği (en iyi beyni) geri yükle!
    agent.load(WEIGHTS_FILE.replace(".weights.h5", "_best.weights.h5"))

    traci.start([
        SUMO_BINARY, "-c", SUMO_CFG_FILE,
        "--no-warnings", "--no-step-log", "--ignore-route-errors"
    ])
    print("[Egitim] SUMO baslatildi.")
    print(f"[Egitim] {EPISODES} bolum x {MAX_STEPS} adim basliyor...")

    episode_rewards = []
    best_avg = -float("inf")

    for episode in range(EPISODES):
        state   = env.reset()
        total_r = 0.0

        for _ in range(MAX_STEPS):
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            total_r += reward
            if DL_AVAILABLE:
                agent.remember(state, action, reward, next_state, done)
                agent.replay()
            else:
                agent.learn(state, action, reward, next_state, done)
            state = next_state
            if done:
                break

        episode_rewards.append(total_r)
        avg = np.mean(episode_rewards[-50:]) if len(episode_rewards) >= 50 else total_r

        if episode % 50 == 0:
            print(f"  Bolum {episode:4d}/{EPISODES} | "
                  f"Odul: {total_r:8.1f} | "
                  f"Ort(50): {avg:8.1f} | "
                  f"eps: {agent.epsilon:.4f}")

        if DL_AVAILABLE and avg > best_avg and episode >= 50:
            best_avg = avg
            agent.model.save_weights(WEIGHTS_FILE.replace(".weights.h5", "_best.weights.h5"))
            if agent.epsilon > EPSILON_END:
                agent.epsilon *= EPSILON_DECAY

    traci.close()
    if DL_AVAILABLE:
        agent.save(WEIGHTS_FILE)
    else:
        agent.save(QTABLE_FILE)
    print("\n[Egitim] Tamamlandi!")
    return agent


def test(initial_route: list[str] | None = None):
    from background_traffic import SUMO_CFG_FILE, SUMO_BINARY
    env   = SUMOEnv(NET_FILE, initial_route)
    agent = DQNAgent() if DL_AVAILABLE else QTableAgent()
    agent.load(WEIGHTS_FILE if DL_AVAILABLE else QTABLE_FILE)

    traci.start([
        SUMO_BINARY, "-c", SUMO_CFG_FILE,
        "--no-warnings", "--no-step-log", "--ignore-route-errors"
    ])
    state, total_r, steps, done = env.reset(), 0.0, 0, False
    while not done and steps < MAX_STEPS:
        action = agent.act(state)
        state, reward, done = env.step(action)
        total_r += reward
        steps   += 1

    traci.close()
    print(f"\n[Test] Adim: {steps} | Toplam Odul: {total_r:.1f}")
    return {"steps": steps, "total_reward": total_r}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RL Ajani — Dinamik Rota")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--test",  action="store_true")
    args = parser.parse_args()
    if args.train:
        train()
    elif args.test:
        test()
    else:
        print("Kullanim: python rl_agent.py --train  veya  --test")