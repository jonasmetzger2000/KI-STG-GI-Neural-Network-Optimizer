from chromosome import Chromosome
from progress import load_generations, write_chromosome, store_old_generations

# Tweaks
population_size = 50
elitists_count = 5

# Variables
populations = load_generations()

# initiale population erzeugen
if len(populations) < 50:
    print("Generiere initiale Bevölkerung..")
    for i in range(population_size):
        populations.append(Chromosome(i))

# Fitness berechnen
for chromosome in populations:
    if chromosome.fitness is None:
        chromosome.determine_fitness()
        write_chromosome(chromosome)
# Current Generation speichern
store_old_generations(populations)
# Selektion (Tournament Selection ohne zurücklegen und Elitist Selektion)
survivors = []

# Eltitists Selektion
elitists = populations[:]
elitists.sort(key=lambda x: x.fitness)
elitists = elitists[0:elitists_count-1]
survivors += elitists
for elitist in elitists:
    populations.remove(elitist)

# Tournament Selektion
it = iter(populations)
for chromosome_a, chromosome_b in zip(it, it):
    populations.remove(chromosome_a)
    populations.remove(chromosome_b)
    survivors.append(chromosome_a if chromosome_a.fitness <= chromosome_b.fitness else chromosome_b)

print(survivors)
print(len(survivors))




