"""
ga_optimizer.py  (v3 — Final)
==============================
Degisiklikler v2 -> v3:
  - allows("passenger") filtresi eklendi (GA artik evrimlesecek)
  - NET_FILE: osm_cleaned.net.xml
"""

import sumolib
import random
import numpy as np

NET_FILE = "osm_cleaned.net.xml"


class GeneticAlgorithmVRP:
    def __init__(
        self,
        net_file: str = NET_FILE,
        num_delivery_points: int = 10,
        population_size: int = 50,
        generations: int = 100,
        use_sumo_routing: bool = False
    ):
        self.net_file         = net_file
        self.num_points       = num_delivery_points
        self.pop_size         = population_size
        self.generations      = generations
        self.use_sumo_routing = use_sumo_routing

        print("Harita yukleniyor...")
        self.net = sumolib.net.readNet(net_file)

        # Kritik filtre: sadece temizlenmis, passenger izinli kenarlar
        self.edges = [
            e for e in self.net.getEdges()
            if not e.getID().startswith(":")
            and e.getLength() > 5.0
            and e.allows("passenger")
        ]
        print(f"  {len(self.edges)} gecerli kenar yüklendi.")

        self.delivery_points = random.sample(
            self.edges, min(self.num_points, len(self.edges))
        )
        print(f"  {len(self.delivery_points)} teslimat noktasi secildi.")

    def create_individual(self) -> list:
        ind = list(self.delivery_points)
        random.shuffle(ind)
        return ind

    def initial_population(self) -> list:
        return [self.create_individual() for _ in range(self.pop_size)]

    def _euclidean(self, a, b) -> float:
        c1 = a.getFromNode().getCoord()
        c2 = b.getFromNode().getCoord()
        return float(np.sqrt((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2))

    def _sumo_distance(self, a, b) -> float:
        try:
            path, cost = self.net.getOptimalPath(fromEdge=a, toEdge=b, fastest=False)
            return cost if cost else self._euclidean(a, b)
        except Exception:
            return self._euclidean(a, b)

    def calculate_fitness(self, individual: list) -> float:
        total = 0.0
        for i in range(len(individual) - 1):
            d = self._sumo_distance(individual[i], individual[i+1]) \
                if self.use_sumo_routing else self._euclidean(individual[i], individual[i+1])
            total += d
        return 1.0 / total if total > 0 else 0.0

    def select_parent(self, population: list) -> list:
        tournament = random.sample(population, min(3, len(population)))
        return max(tournament, key=self.calculate_fitness)

    def ordered_crossover(self, p1: list, p2: list) -> list:
        size       = len(p1)
        start, end = sorted(random.sample(range(size), 2))
        child      = [None] * size
        child[start:end] = p1[start:end]
        p2_idx = 0
        for i in range(size):
            if child[i] is None:
                while p2[p2_idx] in child:
                    p2_idx += 1
                child[i] = p2[p2_idx]
        return child

    def mutate(self, individual: list, mutation_rate: float = 0.1) -> list:
        ind = individual[:]
        for i in range(len(ind)):
            if random.random() < mutation_rate:
                j = random.randrange(len(ind))
                ind[i], ind[j] = ind[j], ind[i]
        return ind

    def run_ga(self) -> list:
        population = self.initial_population()
        prev_best  = None

        for gen in range(self.generations):
            population = sorted(population, key=self.calculate_fitness, reverse=True)
            best   = population[0]
            best_f = self.calculate_fitness(best)
            best_d = 1.0 / best_f if best_f > 0 else float("inf")

            if gen % 10 == 0:
                improved = "" if prev_best is None else \
                    f"  (iyilesme: {prev_best - best_d:+.2f} m)" if best_d < (prev_best or best_d+1) else ""
                print(f"  Jenerasyon {gen:4d} — En kisa mesafe: {best_d:.2f} m{improved}")
                prev_best = best_d

            new_pop = [best]
            while len(new_pop) < self.pop_size:
                p1    = self.select_parent(population)
                p2    = self.select_parent(population)
                child = self.ordered_crossover(p1, p2)
                child = self.mutate(child, 0.1)
                new_pop.append(child)
            population = new_pop

        print("  GA Optimizasyonu tamamlandi!")
        return population[0]


if __name__ == "__main__":
    ga    = GeneticAlgorithmVRP(num_delivery_points=5)
    best  = ga.run_ga()
    best_f = ga.calculate_fitness(best)
    best_d = 1.0 / best_f if best_f > 0 else 0
    print(f"\n--- EN VERIMLI ROTA ({best_d:.2f} m) ---")
    for i, edge in enumerate(best):
        print(f"  {i+1}. Durak: {edge.getID()}")