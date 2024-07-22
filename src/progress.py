from chromosome import Chromosome
import os


def write_chromosome(chromosome):
    file = open('progress.csv', 'a+')
    file.write(chromosome.info)
    file.close()


def store_old_generations(chromosomes):
    file = open('current.txt', 'w')
    for chromosome in chromosomes:
        file.write(chromosome.info)
    file.close()


def load_generations():
    if os.path.isfile("current.txt"):
        file = open('current.txt', 'r')
        chromosomes = []
        for line in file:
            split = str(line).split(";")
            id = split[0]
            batch_size = split[1]
            learning_rate = split[2]
            chromosomes.append(Chromosome(id, batch_size, learning_rate))
        return chromosomes
    return []
