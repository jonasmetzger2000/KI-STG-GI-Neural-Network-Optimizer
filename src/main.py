from chromosome import Chromosome, initial_neurone_size, initial_epochs, initial_learning_rate, initial_batch_size
from progress import load_generations, write_chromosome, store_old_generations

# Tweaks
population_size = 50
elitists_count = 5

# Variables
populations = load_generations()

# initiale population erzeugen
while len(populations) < population_size:
    populations.append(Chromosome(len(populations), initial_learning_rate(), initial_batch_size(), initial_epochs(), initial_neurone_size()))

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
elitists.sort(key=lambda x: x.fitness, reverse=True)
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

survivors.sort(key=lambda x: x.fitness, reverse=True)
print(survivors)
print(len(survivors))




