"""
background_traffic.py  (v3 — Final)
=====================================
Degisiklikler v2 -> v3:
  - NET_FILE: osm_cleaned.net.xml
  - generate_sumo_config: HBEFA3 emisyon modeli aktif edildi
"""

import traci
import random
import sumolib
import os

SUMO_BINARY   = "sumo"
NET_FILE      = "osm_cleaned.net.xml"
ROUTE_FILE    = "background.rou.xml"
SUMO_CFG_FILE = "simulation.sumocfg"

NUM_BACKGROUND_VEHICLES = 300
VEHICLE_SPAWN_STEP      = 5
MAX_STEPS               = 3600


class BackgroundTrafficManager:
    def __init__(self, net_file: str = NET_FILE):
        self.net = sumolib.net.readNet(net_file)
        self.all_edges = [
            e for e in self.net.getEdges()
            if not e.getID().startswith(":") and e.allows("passenger")
        ]
        self._vehicle_counter = 0
        print(f"[Trafik] {len(self.all_edges)} kullanilabilir kenar yuklendi.")

    def _random_edge_pair(self):
        origin = random.choice(self.all_edges)
        dest   = random.choice(self.all_edges)
        while dest.getID() == origin.getID():
            dest = random.choice(self.all_edges)
        return origin, dest

    @staticmethod
    def vehicle_type_xml() -> str:
        return """
    <vType id="car"   accel="2.6" decel="4.5" sigma="0.5" length="5"   maxSpeed="50" color="0.8,0.8,0.8" emissionClass="HBEFA3/PC_G_EU4"/>
    <vType id="truck" accel="1.3" decel="3.5" sigma="0.4" length="12"  maxSpeed="40" color="0.6,0.4,0.2" emissionClass="HBEFA3/HDV_D_EU4"/>
    <vType id="moto"  accel="3.5" decel="5.0" sigma="0.6" length="2.5" maxSpeed="60" color="0.2,0.2,0.8" emissionClass="HBEFA3/PC_G_EU4"/>
"""

    def write_route_file(self, output_path: str = ROUTE_FILE, num_vehicles: int = NUM_BACKGROUND_VEHICLES):
        vehicles_data = []
        for i in range(num_vehicles):
            origin, dest = self._random_edge_pair()
            depart = random.randint(0, 600)
            vtype  = random.choice(["car", "car", "car", "truck", "moto"])
            vehicles_data.append({
                "id": i, "vtype": vtype, "depart": depart,
                "orig": origin.getID(), "dest": dest.getID()
            })
        vehicles_data.sort(key=lambda x: x["depart"])

        lines = ['<routes>\n', self.vehicle_type_xml()]
        for v in vehicles_data:
            lines.append(
                f'    <vehicle id="bg_{v["id"]}" type="{v["vtype"]}" depart="{v["depart"]}">\n'
                f'        <route edges="{v["orig"]} {v["dest"]}"/>\n'
                f'    </vehicle>\n'
            )
        lines.append('</routes>\n')
        with open(output_path, "w", encoding="utf-8") as f:
            f.writelines(lines)
        print(f"[Trafik] {num_vehicles} araclik rota dosyasi yazildi -> {output_path}")

    def spawn_vehicle(self) -> str | None:
        self._vehicle_counter += 1
        veh_id = f"bg_live_{self._vehicle_counter}"
        origin, dest = self._random_edge_pair()
        try:
            traci.route.add(f"route_{veh_id}", [origin.getID()])
            vtype = random.choice(["car", "car", "car", "truck", "moto"])
            traci.vehicle.add(veh_id, f"route_{veh_id}", typeID=vtype)
            traci.vehicle.changeTarget(veh_id, dest.getID())
            return veh_id
        except traci.exceptions.TraCIException as e:
            print(f"[Trafik] Arac eklenemedi: {e}")
            return None

    @staticmethod
    def get_edge_density(edge_id: str) -> int:
        try:
            return traci.edge.getLastStepVehicleNumber(edge_id)
        except traci.exceptions.TraCIException:
            return 0

    @staticmethod
    def get_density_map(edge_ids: list[str]) -> dict[str, int]:
        return {eid: BackgroundTrafficManager.get_edge_density(eid) for eid in edge_ids}


def generate_sumo_config(net_file: str, route_file: str, output: str = SUMO_CFG_FILE):
    """HBEFA3 emisyon modeli aktif SUMO config dosyasi."""
    cfg = f"""<configuration>
  <input>
    <net-file value="{net_file}"/>
    <route-files value="{route_file}"/>
  </input>
  <time>
    <begin value="0"/>
    <end value="{MAX_STEPS}"/>
    <step-length value="1"/>
  </time>
  <emissions>
    <device.emissions.probability value="1.0"/>
  </emissions>
  <report>
    <verbose value="true"/>
    <no-step-log value="true"/>
  </report>
</configuration>
"""
    with open(output, "w") as f:
        f.write(cfg)
    print(f"[Config] SUMO konfigurasyonu yazildi -> {output}")


if __name__ == "__main__":
    manager = BackgroundTrafficManager(NET_FILE)
    manager.write_route_file(ROUTE_FILE, num_vehicles=NUM_BACKGROUND_VEHICLES)
    generate_sumo_config(NET_FILE, ROUTE_FILE)

    traci.start([SUMO_BINARY, "-c", SUMO_CFG_FILE, "--no-warnings", "--ignore-route-errors"])
    print("[SUMO] Simulasyon baslatildi.")

    step = spawned_count = 0
    try:
        while step < MAX_STEPS and traci.simulation.getMinExpectedNumber() > 0:
            traci.simulationStep()
            if step % VEHICLE_SPAWN_STEP == 0 and spawned_count < NUM_BACKGROUND_VEHICLES:
                veh = manager.spawn_vehicle()
                if veh:
                    spawned_count += 1
            if step % 100 == 0:
                active = traci.vehicle.getIDCount()
                print(f"  Adim {step:4d} | Aktif arac: {active:3d} | Uretilen: {spawned_count}")
            step += 1
    finally:
        traci.close()
        print(f"[SUMO] Simulasyon tamamlandi. Toplam adim: {step}")