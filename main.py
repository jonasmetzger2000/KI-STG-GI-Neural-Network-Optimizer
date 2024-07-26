import json
import math
import os
from statistics import median
from random import randrange, uniform, shuffle

import numpy as np


class Gaus:
    def __init__(self, _from, _to, variance_factor):
        self._from = _from
        self._to = _to
        self.variance_factor = variance_factor
        self.rng = np.random.default_rng()

    def compute(self, x1, x2, map=None):
        expectation = median([x1, x2])
        between = max(x1, x2) - min(x1, x2)
        sample = self.rng.normal(expectation, between / self.variance_factor, size=1)
        if map is None:
            return max(self._from, min(self._to, float(sample[0])))
        else:
            return map(max(self._from, min(self._to, float(sample[0]))))


class Chromosome:
    def __init__(self, nr, learning_rate, batch_size, epochs, neurons):
        self.history = None
        self.nr = nr
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.neurons = neurons

    def fitness(self):
        return self.history["val_loss"][-1] if self.history else None

    def __str__(self):
        if self.history is None:
            return "(nr={}, learning_rate={}, batch_size={}, epochs={}, neurons={})".format(
                self.nr,
                self.learning_rate,
                self.batch_size,
                self.epochs,
                self.neurons)
        else:
            return "(nr={}, acc={}, loss={}, learning_rate={}, batch_size={}, epochs={}, neurons={})".format(
                self.nr,
                self.history["val_acc"][-1],
                self.history["val_loss"][-1],
                self.learning_rate,
                self.batch_size,
                self.epochs,
                self.neurons)

    def __repr__(self):
        return "{},{},{},{},{},{},{},{},{}\n".format(
            self.nr,
            self.learning_rate,
            self.batch_size,
            self.epochs,
            self.neurons,
            ";".join(str(x) for x in self.history['acc']),
            ";".join(str(x) for x in self.history['val_acc']),
            ";".join(str(x) for x in self.history['loss']),
            ";".join(str(x) for x in self.history['val_loss']))


def initial_batch_size():
    return randrange(8, 512)


def initial_learning_rate():
    return 10 ** uniform(math.log10(0.00001), math.log10(0.1))


def initial_epochs():
    return randrange(5, 20)


def initial_neurone_size():
    return randrange(10, 1000)


def calculate(chromosome):
    code = os.system("python run.py {} {} {} {} > /dev/null 2>&1".format(
        chromosome.learning_rate,
        chromosome.batch_size,
        chromosome.epochs,
        chromosome.neurons))
    if code == 2:
        exit(2)

    file = open('temp', 'r')
    chromosome.history = json.loads(file.readline())
    print(str(chromosome))


def save_population(chromosomes):
    gen = 0
    while os.path.isfile("current" + str(gen) + ".csv"):
        gen += 1
    file = open("current" + str(gen) + ".csv", 'w')
    for gene in chromosomes:
        file.write(repr(gene))
    file.close()


def get_population(population_size):
    population = []
    gen = 0
    if not os.path.isfile("current" + str(gen) + ".csv"):
        # Erstelle initial Population
        while len(population) < population_size:
            chromosome = Chromosome(len(population),
                                    initial_learning_rate(),
                                    initial_batch_size(),
                                    initial_epochs(),
                                    initial_neurone_size())
            print(str(chromosome))
            population.append(chromosome)
        return population
    # Benutze bereits bestehende Generation
    while not os.path.isfile("current" + str(gen) + ".csv"):
        gen += 1
    file = open("current" + str(gen) + ".csv", 'r')
    for line in file:
        split = str(line).split(";")
        population.append(Chromosome(int(split[0]), float(split[1]), int(split[2]), int(split[3]), int(split[4])))
    file.close()
    return population


def genetic_algorithm():
    # Tweaks
    population_size = 5
    elitists_count = 1
    mutation_rate = 0.20

    population = get_population(population_size)

    while True:
        # Fitness berechnen
        print("### Fitness Berechnen ###")
        for chromosome in population:
            calculate(chromosome)
        save_population(population)
        # Selektion (Tournament Selection ohne zurücklegen und Elitist Selektion)
        survivors = []

        # Eltitists Selektion
        print("### Elitist")
        elitists = population[:]
        elitists.sort(key=lambda x: x.fitness())
        elitists = elitists[0:elitists_count]
        for elitist in elitists:
            print(str(elitist))
            population.remove(elitist)

        # Tournament Selektion
        def pairwise(iterable):
            a = iter(iterable)
            return zip(a, a)

        for chromosome_a, chromosome_b in pairwise(population):
            survivors.append(chromosome_a if chromosome_a.fitness() <= chromosome_b.fitness() else chromosome_b)

        survivors.sort(key=lambda x: x.fitness(), reverse=True)
        population = survivors

        # Rekombination
        print("### Rekombination ###")
        next_id = max(population, key=lambda x: x.nr).nr + 1
        while len(population) + elitists_count < population_size:
            shuffle(population)
            parent1 = population[0]
            parent2 = population[1]
            id = next_id
            next_id += 1
            learning_rate = Gaus(0.00001, 0.1, 3).compute(parent1.learning_rate, parent2.learning_rate)
            batch_size = Gaus(8, 512, 3).compute(parent1.batch_size, parent2.batch_size, lambda x: int(x))
            epochs = Gaus(5, 20, 3).compute(parent1.epochs, parent2.epochs, lambda x: int(x))
            neurons = Gaus(10, 1000, 3).compute(parent1.neurons, parent2.neurons, lambda x: int(x))
            child = Chromosome(id, learning_rate, batch_size, epochs, neurons)
            print(str(child))
            population.append(child)

        # Mutation
        print("### Mutation ###")
        for chromosome in population:
            # learning rate mutieren
            if uniform(0, 1) <= mutation_rate:
                chromosome.history = None
                chromosome.learning_rate = initial_learning_rate()
            # batch size mutieren
            if uniform(0, 1) <= mutation_rate:
                chromosome.history = None
                chromosome.batch_size = initial_batch_size()
            # epochs mutieren
            if uniform(0, 1) <= mutation_rate:
                chromosome.history = None
                chromosome.epochs = initial_epochs()
            # neurons mutieren
            if uniform(0, 1) <= mutation_rate:
                chromosome.history = None
                chromosome.neurons = initial_neurone_size()
            print(str(chromosome))

        for elitist in elitists:
            population.append(elitist)


genetic_algorithm()
