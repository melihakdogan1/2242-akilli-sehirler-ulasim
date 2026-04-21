"""
ga_optimizer.py  (v2 — Hibrit Sistem için güncellendi)
=======================================================
Değişiklikler v1 → v2:
  - Kuş uçuşu mesafe yerine SUMO ağırlıklı Dijkstra desteği
  - Edge geçerlilik kontrolü (tek yönlü, ölü uç filtresi)
  - Sonuç olarak edge ID listesi (hibrit_system.py ile uyumlu)
"""

import sumolib
import random
import numpy as np


class GeneticAlgorithmVRP:
    def __init__(
        self,
        net_file: str,
        num_delivery_points: int = 10,
        population_size: int = 50,
        generations: int = 100,
        use_sumo_routing: bool = False   # True: daha yavaş ama gerçek mesafe
    ):
        self.net_file          = net_file
        self.num_points        = num_delivery_points
        self.pop_size          = population_size
        self.generations       = generations
        self.use_sumo_routing  = use_sumo_routing

        print("Harita yükleniyor, bu birkaç saniye sürebilir...")
        self.net = sumolib.net.readNet(net_file)

        # Geçerli kenarları filtrele: uzunluğu > 5m, kavşak iç kenarı değil
        all_edges = self.net.getEdges()
        self.edges = [
            e for e in all_edges
            if not e.getID().startswith(":")
            and e.getLength() > 5.0
        ]

        self.delivery_points = random.sample(self.edges, min(self.num_points, len(self.edges)))
        print(f"{len(self.delivery_points)} adet teslimat noktası belirlendi.")

    # ------------------------------------------------------------------
    def create_individual(self) -> list:
        individual = list(self.delivery_points)
        random.shuffle(individual)
        return individual

    def initial_population(self) -> list:
        return [self.create_individual() for _ in range(self.pop_size)]

    # ------------------------------------------------------------------
    def _euclidean(self, edge_a, edge_b) -> float:
        """Kenarların başlangıç düğümleri arasındaki Öklid mesafesi."""
        c1 = edge_a.getFromNode().getCoord()
        c2 = edge_b.getFromNode().getCoord()
        return float(np.sqrt((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2))

    def _sumo_distance(self, edge_a, edge_b) -> float:
        """
        SUMO'nun ağırlıklı Dijkstra'sı ile gerçek yol mesafesi.
        Çok yavaş — sadece küçük rotalarda kullan.
        """
        try:
            path, cost = self.net.getOptimalPath(
                fromEdge = edge_a,
                toEdge   = edge_b,
                fastest  = False   # En kısa mesafe
            )
            return cost if cost else self._euclidean(edge_a, edge_b)
        except Exception:
            return self._euclidean(edge_a, edge_b)

    def calculate_fitness(self, individual: list) -> float:
        """Uygunluk = 1 / toplam mesafe."""
        total = 0.0
        for i in range(len(individual) - 1):
            if self.use_sumo_routing:
                d = self._sumo_distance(individual[i], individual[i+1])
            else:
                d = self._euclidean(individual[i], individual[i+1])
            total += d
        return 1.0 / total if total > 0 else 0.0

    # ------------------------------------------------------------------
    def select_parent(self, population: list) -> list:
        """Turnuva seçimi (k=3)."""
        tournament = random.sample(population, min(3, len(population)))
        return max(tournament, key=self.calculate_fitness)

    def ordered_crossover(self, parent1: list, parent2: list) -> list:
        """OX Sıralı Çaprazlama — yineleme olmadan çocuk üretir."""
        size       = len(parent1)
        start, end = sorted(random.sample(range(size), 2))
        child      = [None] * size
        child[start:end] = parent1[start:end]
        p2_idx = 0
        for i in range(size):
            if child[i] is None:
                while parent2[p2_idx] in child:
                    p2_idx += 1
                child[i] = parent2[p2_idx]
        return child

    def mutate(self, individual: list, mutation_rate: float = 0.1) -> list:
        """Swap Mutasyonu — %10 ihtimalle iki durağı değiştir."""
        ind = individual[:]
        for i in range(len(ind)):
            if random.random() < mutation_rate:
                j        = random.randrange(len(ind))
                ind[i], ind[j] = ind[j], ind[i]
        return ind

    # ------------------------------------------------------------------
    def run_ga(self) -> list:
        """Evrim döngüsü. En iyi bireyi (rota) döndürür."""
        population = self.initial_population()

        for gen in range(self.generations):
            population = sorted(population, key=self.calculate_fitness, reverse=True)

            best     = population[0]
            best_f   = self.calculate_fitness(best)
            best_d   = 1.0 / best_f if best_f > 0 else float("inf")

            if gen % 10 == 0:
                print(f"  Jenerasyon {gen:4d} — En kısa mesafe: {best_d:.2f} m")

            # Elitizm + yeni nesil
            new_pop = [best]
            while len(new_pop) < self.pop_size:
                p1    = self.select_parent(population)
                p2    = self.select_parent(population)
                child = self.ordered_crossover(p1, p2)
                child = self.mutate(child, 0.1)
                new_pop.append(child)
            population = new_pop

        print("  GA Optimizasyonu tamamlandı!")
        return population[0]


# -------------------------------------------------------------------
if __name__ == "__main__":
    ga        = GeneticAlgorithmVRP("osm.net.xml.gz", num_delivery_points=5)
    best      = ga.run_ga()
    best_f    = ga.calculate_fitness(best)
    best_dist = 1.0 / best_f if best_f > 0 else 0

    print(f"\n--- EN VERİMLİ ROTA ({best_dist:.2f} m) ---")
    for i, edge in enumerate(best):
        print(f"  {i+1}. Durak: {edge.getID()}")