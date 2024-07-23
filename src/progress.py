from chromosome import Chromosome
import os


def write_chromosome(chromosome):
    file = open('progress.csv', 'a+')
    file.write(chromosome.info)
    file.close()


def store_old_generations(chromosomes):
    file = open('current.txt', 'w')
    for chromosome in chromosomes:
        if chromosome.info is not None:
            file.write(chromosome.info)
    file.close()


def load_generations():
    if os.path.isfile("current.txt"):
        file = open('current.txt', 'r')
        chromosomes = []
        for line in file:
            split = str(line).split(";")
            id = split[0]
            fitness = split[1]
            batch_size = int(split[2])
            learning_rate = float(split[3])
            epochs = int(split[4])
            neurons = int(split[5])
            info = line
            chromosomes.append(Chromosome(id, batch_size, learning_rate, epochs, neurons, fitness, info))
        return chromosomes
    return []
