import sumolib
import random
import numpy as np

class GeneticAlgorithmVRP:
    def __init__(self, net_file, num_delivery_points=10, population_size=50, generations=100):
        self.net_file = net_file
        self.num_points = num_delivery_points
        self.pop_size = population_size
        self.generations = generations
        
        # SUMO haritasını (osm.net.xml) Python'a yüklüyoruz
        print("Harita yükleniyor, bu birkaç saniye sürebilir...")
        self.net = sumolib.net.readNet(net_file)
        
        # Haritadan rastgele teslimat noktaları (edge'ler) seçiyoruz
        self.edges = self.net.getEdges()
        self.delivery_points = random.sample(self.edges, self.num_points)
        print(f"{self.num_points} adet teslimat noktası belirlendi.")

    def create_individual(self):
        """
        Kromozom (Birey) Oluşturma:
        Teslimat noktalarının rastgele sıralanmış hali bizim bir rotamızı (kromozom) temsil eder.
        """
        individual = list(self.delivery_points)
        random.shuffle(individual)
        return individual

    def initial_population(self):
        """Başlangıç popülasyonunu oluşturur."""
        return [self.create_individual() for _ in range(self.pop_size)]

    def calculate_fitness(self, individual):
        """
        Uygunluk (Fitness) Fonksiyonu:
        Literatürdeki 'Neighborhood 2' mantığına göre rotanın toplam uzunluğunu hesaplar.
        Amaç: Toplam mesafeyi (ve dolaylı olarak CO2 emisyonunu) minimize etmek.
        """
        total_distance = 0.0
        for i in range(len(individual) - 1):
            edge_start = individual[i]
            edge_end = individual[i+1]
            
            # Basit mesafe hesaplaması (Kuş uçuşu)
            # İleride buraya SUMO'nun gerçek yol rotalama algoritmasını (Dijkstra) ekleyeceğiz.
            coord1 = edge_start.getFromNode().getCoord()
            coord2 = edge_end.getFromNode().getCoord()
            
            distance = np.sqrt((coord1[0] - coord2[0])**2 + (coord1[1] - coord2[1])**2)
            total_distance += distance
            
        # Mesafe ne kadar kısaysa, fitness o kadar yüksek olmalı (1 / mesafe)
        return 1.0 / float(total_distance) if total_distance > 0 else 0

    def run_ga(self):
        """Genetik Algoritmanın Ana Döngüsü"""
        population = self.initial_population()
        
        for generation in range(self.generations):
            # Fitness değerlerine göre popülasyonu değerlendir
            population = sorted(population, key=self.calculate_fitness, reverse=True)
            
            best_fitness = self.calculate_fitness(population[0])
            if generation % 10 == 0:
                print(f"Jenerasyon {generation} - En İyi Fitness (Skor): {best_fitness:.5f}")
            
            # --- BURAYA ÇAPRAZLANMA (CROSSOVER) VE MUTASYON EKLENECEK ---
            # Şimdilik popülasyonu sabit tutuyoruz.
            
        print("GA Optimizasyonu Tamamlandı!")
        return population[0] # En iyi rotayı döndür

# Test için (Eğer osm.net.xml dosyan hazırsa bu dosya tek başına çalışır)
if __name__ == "__main__":
    # Harita indikten sonra oluşan .net.xml.gz dosyasının adını buraya yazacağız
    ga = GeneticAlgorithmVRP("osm.net.xml.gz", num_delivery_points=5)
    best_route = ga.run_ga()
    print("\n--- EN VERİMLİ ROTA ---")
    # Her bir durağın (Edge) ID'sini yazdırıyoruz
    for i, edge in enumerate(best_route):
        print(f"{i+1}. Durak: {edge.getID()}")