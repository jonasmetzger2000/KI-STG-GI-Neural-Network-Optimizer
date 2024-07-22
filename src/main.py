from chromosome import Chromosome
from progress import load_generations, write_chromosome, store_old_generations

# Tweaks
population_size = 50

# Variables
populations = load_generations()

# initiale population erzeugen
if len(populations) < 50:
    print("Generiere initiale Bevölkerung..")
    for i in range(population_size):
        populations.append(Chromosome(i))

# Fitness berechnen
for chromosome in populations:
    chromosome.determine_fitness()
    write_chromosome(chromosome)
# Current Generation speichern
store_old_generations(populations)
# Selektion (Tournament Selection ohne zurücklegen und Elitist Selektion)
elitists = populations[:]
elitists.sort(key=lambda x: x.fitness)

survivors = []
survivors += elitists
print(populations)
print(elitists)


