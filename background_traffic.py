"""
background_traffic.py
=====================
SUMO simülasyonuna arka plan trafiği ekler.
TraCI üzerinden araç üretir — randomTrips.py'ye gerek kalmaz.

Kullanım:
    Doğrudan çalıştırmak için:
        python background_traffic.py

    Veya diğer modüllerden import ederek:
        from background_traffic import BackgroundTrafficManager
"""

import traci
import random
import sumolib
import os
import subprocess
import sys

# -------------------------------------------------------------------
# YAPILANDIRMA
# -------------------------------------------------------------------
SUMO_BINARY   = "sumo"          # Arayüzsüz çalıştırmak için: "sumo"
NET_FILE      = "osm.net.xml.gz"
ROUTE_FILE    = "background.rou.xml"
SUMO_CFG_FILE = "simulation.sumocfg"

NUM_BACKGROUND_VEHICLES = 300       # Arka plan araç sayısı
VEHICLE_SPAWN_STEP      = 5         # Her kaç adımda bir araç üret
MAX_STEPS               = 3600      # Simülasyon adımı (1 adım = 1 saniye)


class BackgroundTrafficManager:
    """
    TraCI bağlantısı üzerinden arka plan trafiğini yönetir.
    GA/RL modülleriyle birlikte kullanılmak üzere tasarlanmıştır.
    """

    def __init__(self, net_file: str = NET_FILE):
        self.net = sumolib.net.readNet(net_file)
        self.all_edges = [
            e for e in self.net.getEdges()
            # Kavşak içlerini alma VE sadece binek araçların (passenger) geçebildiği yolları al!
            if not e.getID().startswith(":") and e.allows("passenger")
        ]
        self._vehicle_counter = 0
        print(f"[Trafik] {len(self.all_edges)} kullanılabilir kenar yüklendi.")

    # ------------------------------------------------------------------
    # Yardımcı: Rastgele geçerli kenar çifti seç
    # ------------------------------------------------------------------
    def _random_edge_pair(self):
        """Başlangıç ve bitiş için birbirinden farklı iki kenar döndürür."""
        origin      = random.choice(self.all_edges)
        destination = random.choice(self.all_edges)
        while destination.getID() == origin.getID():
            destination = random.choice(self.all_edges)
        return origin, destination

    # ------------------------------------------------------------------
    # Araç tipi tanımları (SUMO route dosyasına eklenecek)
    # ------------------------------------------------------------------
    @staticmethod
    def vehicle_type_xml() -> str:
        return """
    <vType id="car"    accel="2.6" decel="4.5" sigma="0.5" length="5"   maxSpeed="50" color="0.8,0.8,0.8"/>
    <vType id="truck"  accel="1.3" decel="3.5" sigma="0.4" length="12"  maxSpeed="40" color="0.6,0.4,0.2"/>
    <vType id="moto"   accel="3.5" decel="5.0" sigma="0.6" length="2.5" maxSpeed="60" color="0.2,0.2,0.8"/>
"""

    # ------------------------------------------------------------------
    # SUMO route XML dosyası yaz (ön-ısınma için)
    # ------------------------------------------------------------------
    def write_route_file(self, output_path: str = ROUTE_FILE, num_vehicles: int = 50):
        """
        Simülasyon başlamadan önce belirli sayıda araç rotasını
        XML olarak yazar. Kalkış saatlerine göre sıralanmıştır.
        """
        # Araçları önce bir sözlük listesinde toplayalım ki sıralayabilelim
        vehicles_data = []
        for i in range(num_vehicles):
            origin, dest = self._random_edge_pair()
            depart = random.randint(0, 600)          # İlk 10 dakikaya yay
            vtype  = random.choice(["car", "car", "car", "truck", "moto"])
            vehicles_data.append({
                "id": i, "vtype": vtype, "depart": depart,
                "orig": origin.getID(), "dest": dest.getID()
            })
            
        # Araçları kalkış saatine (depart) göre küçükten büyüğe sırala! (Uyarıları çözer)
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
        print(f"[Trafik] {num_vehicles} araçlık sıralı rota dosyası yazıldı → {output_path}")

    # ------------------------------------------------------------------
    # TraCI üzerinden CANLI araç ekle (simülasyon çalışırken çağrılır)
    # ------------------------------------------------------------------
    def spawn_vehicle(self) -> str | None:
        """
        Simülasyon adımı sırasında rastgele bir araca kaynak verir.
        Geçerli bir rota bulunamazsa None döner.
        """
        self._vehicle_counter += 1
        veh_id = f"bg_live_{self._vehicle_counter}"
        origin, dest = self._random_edge_pair()

        try:
            # SUMO'nun en kısa yolu kendisi hesaplamasi için boş rota ekle
            traci.route.add(f"route_{veh_id}", [origin.getID()])
            vtype = random.choice(["car", "car", "car", "truck", "moto"])
            traci.vehicle.add(veh_id, f"route_{veh_id}", typeID=vtype)
            # Hedefe gitmesini söyle (TraCI rerouting)
            traci.vehicle.changeTarget(veh_id, dest.getID())
            return veh_id
        except traci.exceptions.TraCIException as e:
            print(f"[Trafik] Araç eklenemedi: {e}")
            return None

    # ------------------------------------------------------------------
    # Belirli bir kenardaki araç yoğunluğunu ölç
    # ------------------------------------------------------------------
    @staticmethod
    def get_edge_density(edge_id: str) -> int:
        """Verilen kenardaki anlık araç sayısını döndürür."""
        try:
            return traci.edge.getLastStepVehicleNumber(edge_id)
        except traci.exceptions.TraCIException:
            return 0

    # ------------------------------------------------------------------
    # Tüm yakın kenarların yoğunluk haritasını döndür
    # ------------------------------------------------------------------
    @staticmethod
    def get_density_map(edge_ids: list[str]) -> dict[str, int]:
        """Birden fazla kenar için {edge_id: araç_sayısı} sözlüğü döndürür."""
        return {eid: BackgroundTrafficManager.get_edge_density(eid) for eid in edge_ids}


# -------------------------------------------------------------------
# BAĞIMSIZ TEST MODU
# -------------------------------------------------------------------
def generate_sumo_config(net_file: str, route_file: str, output: str = SUMO_CFG_FILE):
    """Basit bir .sumocfg dosyası oluşturur."""
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
  <report>
    <verbose value="true"/>
    <no-step-log value="true"/>
  </report>
</configuration>
"""
    with open(output, "w") as f:
        f.write(cfg)
    print(f"[Config] SUMO konfigürasyonu yazıldı → {output}")


if __name__ == "__main__":
    # 1) Trafik dosyalarını oluştur
    manager = BackgroundTrafficManager(NET_FILE)
    manager.write_route_file(ROUTE_FILE, num_vehicles=NUM_BACKGROUND_VEHICLES)
    generate_sumo_config(NET_FILE, ROUTE_FILE)

    # 2) SUMO + TraCI bağlantısını başlat
    traci.start([SUMO_BINARY, "-c", SUMO_CFG_FILE, "--no-warnings"])
    print("[SUMO] Simülasyon başlatıldı.")

    step            = 0
    spawned_count   = 0

    try:
        while step < MAX_STEPS and traci.simulation.getMinExpectedNumber() > 0:
            traci.simulationStep()

            # Her VEHICLE_SPAWN_STEP adımda yeni araç ekle
            if step % VEHICLE_SPAWN_STEP == 0 and spawned_count < NUM_BACKGROUND_VEHICLES:
                veh = manager.spawn_vehicle()
                if veh:
                    spawned_count += 1

            # Her 100 adımda durum özeti
            if step % 100 == 0:
                active = traci.vehicle.getIDCount()
                print(f"  Adım {step:4d} | Aktif araç: {active:3d} | Toplam üretilen: {spawned_count}")

            step += 1

    finally:
        traci.close()
        print(f"[SUMO] Simülasyon tamamlandı. Toplam adım: {step}")