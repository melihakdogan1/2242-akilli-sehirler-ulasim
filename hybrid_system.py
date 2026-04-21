"""
hybrid_system.py
================
GA + RL Hibrit Sisteminin Ana Orkestratörü.

Sabah başlatma akışı:
  1. GA çalışır → optimum başlangıç rotasını hesaplar
  2. SUMO simülasyonu başlatılır + arka plan trafiği eklenir
  3. Araç GA rotasıyla yola çıkar
  4. TraCI sensörü önündeki kenar yoğunluğunu izler
  5. Yoğunluk eşiği aşılırsa → RL ajanı devreye girer
  6. Rota yeniden hesaplanır, araç hedefe ulaşır
  7. Metrikler (mesafe, CO₂, süre) raporlanır

Kullanım:
    python hybrid_system.py
"""

import os
import time
import numpy as np
import traci

from ga_optimizer      import GeneticAlgorithmVRP
from background_traffic import BackgroundTrafficManager, generate_sumo_config, SUMO_BINARY
from rl_agent           import SUMOEnv, DQNAgent, QTableAgent, DL_AVAILABLE

# -------------------------------------------------------------------
# YAPILANDIRMA
# -------------------------------------------------------------------
NET_FILE         = "osm.net.xml.gz"
ROUTE_FILE       = "background.rou.xml"
SUMO_CFG_FILE    = "simulation.sumocfg"

# GA parametreleri
GA_DELIVERY_PTS  = 5
GA_POPULATION    = 50
GA_GENERATIONS   = 100

# Hibrit eşik
REROUTE_DENSITY  = 2    # Önündeki kenarda bu kadar araç varsa RL devreye girer

# Simülasyon
MAX_SIM_STEPS    = 1800  # 30 dakikalık simülasyon

AGENT_VEHICLE_ID = "delivery_vehicle"
AGENT_VTYPE      = "delivery_car"


# ===================================================================
# YARDIMCI: CO₂ tahmini
# ===================================================================
def estimate_co2(distance_m: float, avg_speed_kmh: float, stop_count: int) -> float:
    """
    Basitleştirilmiş COPERT modeli.
    Dur-kalk +15%, düşük hız +20% emisyon artışı.
    """
    base_g_per_km = 120.0          # Ortalama benzinli araç: 120 g CO₂/km
    stop_penalty  = 1.0 + (stop_count * 0.05)
    speed_factor  = 1.2 if avg_speed_kmh < 20 else 1.0
    return (distance_m / 1000.0) * base_g_per_km * stop_penalty * speed_factor


# ===================================================================
# ADIM 1: GA ile statik rota hesapla
# ===================================================================
def run_ga_phase() -> tuple[list[str], float]:
    """
    GA çalıştırır, en iyi rotayı ve tahmini mesafeyi döndürür.
    Returns: (edge_id_listesi, mesafe_metre)
    """
    print("\n" + "="*60)
    print("  AŞAMA 1: Genetik Algoritma — Başlangıç Rotası")
    print("="*60)

    ga = GeneticAlgorithmVRP(
        net_file          = NET_FILE,
        num_delivery_points = GA_DELIVERY_PTS,
        population_size   = GA_POPULATION,
        generations       = GA_GENERATIONS
    )
    best_route = ga.run_ga()

    edge_ids = [edge.getID() for edge in best_route]
    # Fitness → mesafe dönüşümü
    fitness  = ga.calculate_fitness(best_route)
    distance = 1.0 / fitness if fitness > 0 else 9999.0

    print(f"\n  GA Sonucu: {len(edge_ids)} durak | Tahmini mesafe: {distance:.1f} m")
    print("  Rota:", " → ".join(edge_ids))
    return edge_ids, distance


# ===================================================================
# ADIM 2: SUMO ortamını hazırla
# ===================================================================
def setup_simulation(ga_route: list[str]) -> BackgroundTrafficManager:
    """
    Arka plan trafiğini ve SUMO config dosyasını hazırlar.
    """
    print("\n" + "="*60)
    print("  AŞAMA 2: Simülasyon Ortamı Hazırlanıyor")
    print("="*60)

    manager = BackgroundTrafficManager(NET_FILE)
    manager.write_route_file(ROUTE_FILE, num_vehicles=100)
    generate_sumo_config(NET_FILE, ROUTE_FILE, SUMO_CFG_FILE)

    print("  Arka plan trafiği ve config hazır.")
    return manager


# ===================================================================
# ADIM 3: RL ajanını yükle
# ===================================================================
def load_rl_agent():
    """Eğitilmiş RL ajanını yükler (yoksa yeni başlatır)."""
    print("\n" + "="*60)
    print("  AŞAMA 3: RL Ajanı Yükleniyor")
    print("="*60)

    agent = DQNAgent() if DL_AVAILABLE else QTableAgent()

    weight_file = "dqn_weights.weights.h5" if DL_AVAILABLE else "q_table.pkl"
    if os.path.exists(weight_file):
        agent.load(weight_file)
        print("  Eğitilmiş ağırlıklar yüklendi.")
    else:
        print("  Uyarı: Kayıtlı ağırlık bulunamadı. Keşif modu aktif.")
        print("  İpucu: Önce 'python rl_agent.py --train' komutunu çalıştırın.")

    return agent


# ===================================================================
# ADIM 4: Hibrit simülasyon döngüsü
# ===================================================================
def run_hybrid_loop(
    ga_route: list[str],
    rl_agent,
    traffic_manager: BackgroundTrafficManager
) -> dict:
    """
    Ana simülasyon döngüsü.
    GA rotasını takip eder, trafik algılayınca RL'e devreder.
    """
    print("\n" + "="*60)
    print("  AŞAMA 4: Hibrit Simülasyon Döngüsü")
    import sumolib
    net = sumolib.net.readNet(NET_FILE) # Haritayı SADECE BİR KERE belleğe alıyoruz
    print("="*60)

    # SUMO'yu başlat
    traci.start([SUMO_BINARY, "-c", SUMO_CFG_FILE, "--no-warnings", "--no-step-log", "--ignore-route-errors"])

    # Teslimat aracını ekle
    try:
        traci.vehicletype.copy("car", AGENT_VTYPE)
        traci.vehicletype.setColor(AGENT_VTYPE, (255, 100, 0, 255))  # Turuncu
        traci.route.add("ga_route_init", [ga_route[0]])
        traci.vehicle.add(AGENT_VEHICLE_ID, "ga_route_init", typeID=AGENT_VTYPE)
        traci.vehicle.changeTarget(AGENT_VEHICLE_ID, ga_route[-1])
        traci.simulationStep()
        print(f"  Teslimat aracı eklendi: {AGENT_VEHICLE_ID}")
    except traci.exceptions.TraCIException as e:
        print(f"  Hata: {e}")
        traci.close()
        return {}

    # Metrik takip
    metrics = {
        "ga_phase_steps":  0,
        "rl_phase_steps":  0,
        "reroute_count":   0,
        "stop_count":      0,
        "total_distance":  0.0,
        "mode_log":        [],        # [(adım, mod, kenar)]
        "goal_reached":    False,
    }

    current_mode    = "GA"           # "GA" veya "RL"
    prev_pos        = None
    ga_waypoint_idx = 0              # GA rotasındaki mevcut hedef indeksi
    start_time      = time.time()

    # Arka plan araçlarını ek wave ile ekle
    spawned_bg = 0

    for step in range(MAX_SIM_STEPS):
        traci.simulationStep()

        # Arka plan araç üretimi
        if step % 8 == 0 and spawned_bg < 80:
            traffic_manager.spawn_vehicle()
            spawned_bg += 1

        # Araç simülasyondan çıktıysa
        if AGENT_VEHICLE_ID not in traci.vehicle.getIDList():
            metrics["goal_reached"] = True
            print(f"  Adım {step}: Araç hedefe ulaştı veya simülasyondan çıktı.")
            break

        current_edge = traci.vehicle.getRoadID(AGENT_VEHICLE_ID)
        current_pos  = traci.vehicle.getPosition(AGENT_VEHICLE_ID)
        current_speed = traci.vehicle.getSpeed(AGENT_VEHICLE_ID)

        # Mesafe birikimi
        if prev_pos:
            dx = current_pos[0] - prev_pos[0]
            dy = current_pos[1] - prev_pos[1]
            metrics["total_distance"] += np.sqrt(dx*dx + dy*dy)
        prev_pos = current_pos

        # Dur-kalk sayacı
        if current_speed < 0.5:
            metrics["stop_count"] += 1

        # RL tetikleyici: önündeki kenarda yoğunluk kontrolü
        forward_density = traci.edge.getLastStepVehicleNumber(current_edge)

        if forward_density >= REROUTE_DENSITY and current_mode == "GA":
            # GA modundan RL moduna geç
            current_mode = "RL"
            metrics["reroute_count"] += 1
            print(f"  Adım {step}: ⚠ Trafik algılandı ({forward_density} araç) "
                  f"— RL ajanı devreye girdi!")
            metrics["mode_log"].append((step, "RL_START", current_edge))

        elif forward_density < REROUTE_DENSITY // 2 and current_mode == "RL":
            # Trafik azaldı, GA rotasına geri dön
            current_mode = "GA"
            print(f"  Adım {step}: ✓ Trafik azaldı — GA rotasına dönülüyor.")
            metrics["mode_log"].append((step, "GA_RESUME", current_edge))

        # Moda göre kontrol
        if current_mode == "RL":
            # RL kararı al ve uygula (net değişkenini parametre olarak ekledik)
            state  = _build_state(current_edge, current_pos, ga_route[-1], net)
            action = rl_agent.act(state)

            if action == 3:  # Bekle
                traci.vehicle.setSpeed(AGENT_VEHICLE_ID, 0)
            else:
                traci.vehicle.setSpeed(AGENT_VEHICLE_ID, -1)  # Serbest bırak
                traci.vehicle.rerouteTraveltime(AGENT_VEHICLE_ID)
            metrics["rl_phase_steps"] += 1

        else:
            # GA modu: sıradaki waypoint'e yönlendir
            if ga_waypoint_idx < len(ga_route) - 1:
                next_wp = ga_route[ga_waypoint_idx + 1]
                try:
                    traci.vehicle.changeTarget(AGENT_VEHICLE_ID, next_wp)
                except Exception:
                    pass
                if current_edge == next_wp:
                    ga_waypoint_idx += 1
            metrics["ga_phase_steps"] += 1

        metrics["mode_log"].append((step, current_mode, current_edge))

    traci.close()
    metrics["elapsed_seconds"] = time.time() - start_time
    return metrics


# ===================================================================
# YARDIMCI: state vektörü oluştur
# ===================================================================
def _build_state(current_edge: str, pos: tuple, goal_edge_id: str, net) -> np.ndarray:
    """Anlık konumdan RL state vektörü üretir."""
    try:
        ge  = net.getEdge(goal_edge_id)
        gx  = (ge.getFromNode().getCoord()[0] + ge.getToNode().getCoord()[0]) / 2
        gy  = (ge.getFromNode().getCoord()[1] + ge.getToNode().getCoord()[1]) / 2
        d1  = min(traci.edge.getLastStepVehicleNumber(current_edge), 20) / 20.0
        return np.array([pos[0]/5000, pos[1]/5000, gx/5000, gy/5000, d1, 0.0], dtype=np.float32)
    except Exception:
        return np.zeros(6, dtype=np.float32)


# ===================================================================
# ADIM 5: Rapor
# ===================================================================
def print_report(ga_distance: float, metrics: dict):
    """Simülasyon sonuçlarını raporlar."""
    co2 = estimate_co2(
        distance_m  = metrics.get("total_distance", 0),
        avg_speed_kmh = 25.0,
        stop_count  = metrics.get("stop_count", 0)
    )

    print("\n" + "="*60)
    print("  FİNAL RAPOR")
    print("="*60)
    print(f"  GA başlangıç mesafesi         : {ga_distance:.1f} m")
    print(f"  Gerçek kat edilen mesafe       : {metrics.get('total_distance', 0):.1f} m")
    print(f"  Toplam simülasyon adımı        : {metrics.get('ga_phase_steps',0) + metrics.get('rl_phase_steps',0)}")
    print(f"  GA modunda adım                : {metrics.get('ga_phase_steps', 0)}")
    print(f"  RL modunda adım                : {metrics.get('rl_phase_steps', 0)}")
    print(f"  RL devreye girme sayısı        : {metrics.get('reroute_count', 0)}")
    print(f"  Dur-kalk sayısı               : {metrics.get('stop_count', 0)}")
    print(f"  Tahmini CO₂ emisyonu           : {co2:.1f} g")
    print(f"  Hedefe ulaşıldı mı?            : {'EVET ✓' if metrics.get('goal_reached') else 'HAYIR ✗'}")
    print(f"  Gerçek süre                    : {metrics.get('elapsed_seconds',0):.1f} sn")
    print("="*60)

    return {
        "ga_initial_distance": ga_distance,
        "actual_distance"    : metrics.get("total_distance", 0),
        "reroute_count"      : metrics.get("reroute_count", 0),
        "co2_grams"          : co2,
        "goal_reached"       : metrics.get("goal_reached", False)
    }


# ===================================================================
# ANA GİRİŞ NOKTASI
# ===================================================================
if __name__ == "__main__":
    print("\n" + "#"*60)
    print("#  TEKNOFEST — Kentsel Lojistik Hibrit Sistem  #")
    print("#"*60)

    MAX_DENEME = 50
    for deneme in range(MAX_DENEME):
        # 1. GA fazı
        ga_route, ga_dist = run_ga_phase()

        # 2. Simülasyon ortamı hazırla
        traffic_mgr = setup_simulation(ga_route)

        # 3. RL ajanı yükle
        rl_agent = load_rl_agent()

        # 4. Hibrit döngü
        metrics = run_hybrid_loop(ga_route, rl_agent, traffic_mgr)

        # 5. Rapor ve Çıkış
        if metrics:
            final = print_report(ga_dist, metrics)
            break  # Başarılı olduysa döngüyü kır ve programı bitir
        else:
            print(f"\n[Uyarı] Seçilen rotalar arasında fiziksel bağlantı yok (Çıkmaz sokak).")
            print(f"Oto-Tamir Devrede: Yeni duraklarla baştan başlanıyor... (Deneme {deneme+1}/{MAX_DENEME})")