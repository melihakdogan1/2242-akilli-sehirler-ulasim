"""
hybrid_system.py  (v3 — Final)
================================
Degisiklikler v2 -> v3:
  - estimate_co2() kaldirildi; CO2 dogrudan SUMO'nun HBEFA3 modelinden okunuyor
  - _build_state: 8 elemanli (ilerideki 3 kenar yogunlugu dahil)
  - Dinamik trafik esigi: sabit 2 yerine kenar serit sayisina gore hesaplaniyor
  - REROUTE_DENSITY sabiti demo icin 2 kaliyor ama dinamik fonksiyon da mevcut
  - NET_FILE: osm_cleaned.net.xml
"""

import os
import time
import numpy as np
import traci

from ga_optimizer       import GeneticAlgorithmVRP
from background_traffic import BackgroundTrafficManager, generate_sumo_config, SUMO_BINARY
from rl_agent           import SUMOEnv, DQNAgent, QTableAgent, DL_AVAILABLE, STATE_SIZE

NET_FILE      = "osm_cleaned.net.xml"
ROUTE_FILE    = "background.rou.xml"
SUMO_CFG_FILE = "simulation.sumocfg"

GA_DELIVERY_PTS = 5
GA_POPULATION   = 50
GA_GENERATIONS  = 100

REROUTE_DENSITY = 2      # Demo icin dusuk tutuldu (juri sunumu)
MAX_SIM_STEPS   = 1800

AGENT_VEHICLE_ID = "delivery_vehicle"
AGENT_VTYPE      = "delivery_car"


def get_dynamic_threshold(edge_id: str) -> int:
    """
    Kenar serit sayisina gore dinamik trafik esigi.
    Kapasitenin %60'i dolunca RL devreye girer.
    Demo modunda REROUTE_DENSITY kullanilir.
    """
    try:
        lanes = traci.edge.getLaneNumber(edge_id)
        return max(2, int(lanes * 5 * 0.6))
    except Exception:
        return REROUTE_DENSITY


def run_ga_phase() -> tuple[list[str], float]:
    print("\n" + "="*60)
    print("  ASAMA 1: Genetik Algoritma — Baslangic Rotasi")
    print("="*60)

    ga = GeneticAlgorithmVRP(
        net_file            = NET_FILE,
        num_delivery_points = GA_DELIVERY_PTS,
        population_size     = GA_POPULATION,
        generations         = GA_GENERATIONS
    )
    best_route = ga.run_ga()
    edge_ids   = [e.getID() for e in best_route]
    fitness    = ga.calculate_fitness(best_route)
    distance   = 1.0 / fitness if fitness > 0 else 9999.0

    print(f"\n  GA Sonucu: {len(edge_ids)} durak | Tahmini mesafe: {distance:.1f} m")
    print("  Rota:", " -> ".join(edge_ids))
    return edge_ids, distance


def setup_simulation(ga_route: list[str]) -> BackgroundTrafficManager:
    print("\n" + "="*60)
    print("  ASAMA 2: Simulasyon Ortami Hazirlaniyor")
    print("="*60)
    manager = BackgroundTrafficManager(NET_FILE)
    manager.write_route_file(ROUTE_FILE, num_vehicles=100)
    generate_sumo_config(NET_FILE, ROUTE_FILE, SUMO_CFG_FILE)
    print("  Arka plan trafigi ve config hazir.")
    return manager


def load_rl_agent():
    print("\n" + "="*60)
    print("  ASAMA 3: RL Ajani Yukleniyor")
    print("="*60)
    agent = DQNAgent() if DL_AVAILABLE else QTableAgent()
    weight_file = "dqn_weights.weights.h5" if DL_AVAILABLE else "q_table.pkl"
    if os.path.exists(weight_file):
        agent.load(weight_file)
        print("  Egitilmis agirliklar yuklendi.")
    else:
        print("  Uyari: Agirlik bulunamadi. Kendi kendine ogreniyor.")
    return agent


def _build_state(current_edge: str, pos: tuple, goal_edge_id: str, net) -> np.ndarray:
    """
    8 elemanli state vektoru:
    [x, y, gx, gy, d0(simdi), d1(1 ileride), d2(2 ileride), d3(3 ileride)]
    """
    try:
        ge = net.getEdge(goal_edge_id)
        gx = (ge.getFromNode().getCoord()[0] + ge.getToNode().getCoord()[0]) / 2
        gy = (ge.getFromNode().getCoord()[1] + ge.getToNode().getCoord()[1]) / 2
        X_MAX = Y_MAX = 5000.0

        if AGENT_VEHICLE_ID in traci.vehicle.getIDList():
            route_edges = traci.vehicle.getRoute(AGENT_VEHICLE_ID)
            route_idx   = traci.vehicle.getRouteIndex(AGENT_VEHICLE_ID)
        else:
            route_edges, route_idx = [current_edge], 0

        def ahead(offset: int) -> float:
            idx = route_idx + offset
            if 0 <= idx < len(route_edges):
                return min(traci.edge.getLastStepVehicleNumber(route_edges[idx]), 20) / 20.0
            return 0.0

        return np.array([
            pos[0] / X_MAX,
            pos[1] / Y_MAX,
            gx / X_MAX,
            gy / Y_MAX,
            ahead(0),
            ahead(1),
            ahead(2),
            ahead(3),
        ], dtype=np.float32)

    except Exception:
        return np.zeros(STATE_SIZE, dtype=np.float32)


def run_hybrid_loop(
    ga_route: list[str],
    rl_agent,
    traffic_manager: BackgroundTrafficManager
) -> dict:
    print("\n" + "="*60)
    print("  ASAMA 4: Hibrit Simulasyon Dongusu")
    import sumolib
    net = sumolib.net.readNet(NET_FILE)
    print("="*60)

    traci.start([
        SUMO_BINARY, "-c", SUMO_CFG_FILE,
        "--no-warnings", "--no-step-log", "--ignore-route-errors"
    ])

    try:
        traci.vehicletype.copy("car", AGENT_VTYPE)
        traci.vehicletype.setColor(AGENT_VTYPE, (255, 100, 0, 255))
        traci.route.add("ga_route_init", [ga_route[0]])
        traci.vehicle.add(AGENT_VEHICLE_ID, "ga_route_init", typeID=AGENT_VTYPE)
        traci.vehicle.changeTarget(AGENT_VEHICLE_ID, ga_route[-1])
        traci.simulationStep()
        print(f"  Teslimat araci eklendi: {AGENT_VEHICLE_ID}")
    except traci.exceptions.TraCIException as e:
        print(f"  Hata: {e}")
        traci.close()
        return {}

    metrics = {
        "ga_phase_steps": 0,
        "rl_phase_steps": 0,
        "reroute_count":  0,
        "stop_count":     0,
        "total_distance": 0.0,
        "total_co2_mg":   0.0,   # SUMO HBEFA3 modelinden mg cinsinden
        "total_fuel_mg":  0.0,
        "mode_log":       [],
        "goal_reached":   False,
    }

    current_mode    = "GA"
    prev_pos        = None
    ga_waypoint_idx = 0
    start_time      = time.time()
    spawned_bg      = 0

    for step in range(MAX_SIM_STEPS):
        traci.simulationStep()

        if step % 8 == 0 and spawned_bg < 80:
            traffic_manager.spawn_vehicle()
            spawned_bg += 1

        if AGENT_VEHICLE_ID not in traci.vehicle.getIDList():
            metrics["goal_reached"] = True
            print(f"  Adim {step}: Arac hedefe ulasti veya simulasyondan cikti.")
            break

        current_edge  = traci.vehicle.getRoadID(AGENT_VEHICLE_ID)
        current_pos   = traci.vehicle.getPosition(AGENT_VEHICLE_ID)
        current_speed = traci.vehicle.getSpeed(AGENT_VEHICLE_ID)

        # Mesafe birikimi
        if prev_pos:
            dx = current_pos[0] - prev_pos[0]
            dy = current_pos[1] - prev_pos[1]
            metrics["total_distance"] += np.sqrt(dx*dx + dy*dy)
        prev_pos = current_pos

        # Dur-kalk sayaci
        if current_speed < 0.5:
            metrics["stop_count"] += 1

        # CO2 — SUMO HBEFA3 modelinden (mg/s)
        metrics["total_co2_mg"] += traci.vehicle.getCO2Emission(AGENT_VEHICLE_ID)

        # YAKIT — SUMO HBEFA3 modelinden (mg/s)
        metrics["total_fuel_mg"] += traci.vehicle.getFuelConsumption(AGENT_VEHICLE_ID)

        # Trafik yogunlugu
        forward_density = traci.edge.getLastStepVehicleNumber(current_edge)
        # Demo modunda REROUTE_DENSITY, gercek modda dinamik esik:
        # threshold = get_dynamic_threshold(current_edge)
        threshold = REROUTE_DENSITY

        if forward_density >= threshold and current_mode == "GA":
            current_mode = "RL"
            metrics["reroute_count"] += 1
            print(f"  Adim {step}: Trafik algilandi ({forward_density} arac) "
                  f"— RL ajani devreye girdi!")
            metrics["mode_log"].append((step, "RL_START", current_edge))

        elif forward_density < threshold // 2 and current_mode == "RL":
            current_mode = "GA"
            print(f"  Adim {step}: Trafik azaldi — GA rotasina donuluyor.")
            metrics["mode_log"].append((step, "GA_RESUME", current_edge))

        if current_mode == "RL":
            state  = _build_state(current_edge, current_pos, ga_route[-1], net)
            action = rl_agent.act(state)
            if action == 3:
                traci.vehicle.setSpeed(AGENT_VEHICLE_ID, 0)
            else:
                traci.vehicle.setSpeed(AGENT_VEHICLE_ID, -1)
                traci.vehicle.rerouteTraveltime(AGENT_VEHICLE_ID)
            metrics["rl_phase_steps"] += 1
        else:
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


def print_report(ga_distance: float, metrics: dict):
    # CO2: mg -> g donusumu (SUMO HBEFA3 modelinden)
    co2_grams = metrics.get("total_co2_mg", 0.0) / 1000.0

    fuel_grams = metrics.get("total_fuel_mg", 0.0) / 1000.0
    fuel_liters = fuel_grams / 740.0  # 1 Litre benzin ortalama 740 gramdır

    print("\n" + "="*60)
    print("  FINAL RAPOR")
    print("="*60)
    print(f"  GA baslangic mesafesi         : {ga_distance:.1f} m")
    print(f"  Gercek kat edilen mesafe       : {metrics.get('total_distance', 0):.1f} m")
    print(f"  Toplam simulasyon adimi        : "
          f"{metrics.get('ga_phase_steps',0) + metrics.get('rl_phase_steps',0)}")
    print(f"  GA modunda adim                : {metrics.get('ga_phase_steps', 0)}")
    print(f"  RL modunda adim                : {metrics.get('rl_phase_steps', 0)}")
    print(f"  RL devreye girme sayisi        : {metrics.get('reroute_count', 0)}")
    print(f"  Dur-kalk sayisi               : {metrics.get('stop_count', 0)}")
    print(f"  CO2 emisyonu (SUMO HBEFA3)    : {co2_grams:.1f} g")
    print(f"  Harcanan Yakıt (HBEFA3)       : {fuel_liters:.3f} Litre")
    print(f"  Hedefe ulasildi mi?            : "
          f"{'EVET' if metrics.get('goal_reached') else 'HAYIR'}")
    print(f"  Gercek sure                    : {metrics.get('elapsed_seconds',0):.1f} sn")
    print("="*60)
    return {
        "ga_initial_distance": ga_distance,
        "actual_distance":     metrics.get("total_distance", 0),
        "reroute_count":       metrics.get("reroute_count", 0),
        "co2_grams":           co2_grams,
        "goal_reached":        metrics.get("goal_reached", False)
    }


if __name__ == "__main__":
    print("\n" + "#"*60)
    print("#  TEKNOFEST — Kentsel Lojistik Hibrit Sistem  #")
    print("#"*60)

    MAX_DENEME = 50
    for deneme in range(MAX_DENEME):
        ga_route, ga_dist = run_ga_phase()
        traffic_mgr       = setup_simulation(ga_route)
        rl_agent          = load_rl_agent()
        metrics           = run_hybrid_loop(ga_route, rl_agent, traffic_mgr)

        if metrics:
            print_report(ga_dist, metrics)
            break
        else:
            print(f"\n[Uyari] Fiziksel baglanti yok. "
                  f"Yeni noktalarla yeniden baslanıyor... "
                  f"(Deneme {deneme+1}/{MAX_DENEME})")