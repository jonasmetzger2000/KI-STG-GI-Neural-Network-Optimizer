from statistics import median
import os
import math
from random import randrange, uniform, shuffle
from math import trunc
import time
import keras
import numpy as np
from tensorflow import keras
import tensorflow as tf
import tracemalloc

tracemalloc.start()

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


def execute(chromosome):
    keras.backend.clear_session()
    start_time = time.time()
    model = keras.Sequential([
        keras.layers.Dense(chromosome.neurons, activation='softmax'),
        keras.layers.Flatten(),
        keras.layers.Dense(10, activation='softmax'),
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=float(chromosome.learning_rate)),
        loss='categorical_crossentropy',
        metrics=['acc'])
    history = model.fit(x_train,
                        y_train,
                        int(chromosome.batch_size),
                        epochs=chromosome.epochs,
                        verbose=0,
                        validation_data=(x_test, y_test))
    return history, (time.time() - start_time)


def initial_batch_size():
    return randrange(8, 512)


def initial_learning_rate():
    return 10 ** uniform(math.log10(0.00001), math.log10(0.1))


def initial_epochs():
    return randrange(5, 20)


def initial_neurone_size():
    return randrange(10, 1000)


class Chromosome:
    def __init__(self, id, lr, bs, ep, ne, fitness = None, info = None):
        self.result = None
        self.fitness = None
        self.duration = None
        self.info = info
        self.id = id
        self.learning_rate = lr
        self.batch_size = bs
        self.epochs = ep
        self.neurons = ne
        self.fitness = fitness

    def __copy__(self):
        return Chromosome(self.id, self.learning_rate, self.batch_size, self.epochs, self.neurons, self.fitness, self.info)

    def __str__(self):
        return "(id={}, acc={}, loss={}, seconds={}, batchSize={}, epochs={}, neurons={}, learningRate={})".format(
            self.id,
            self.result.history["val_acc"][-1] if self.result is not None else None,
            self.result.history["val_loss"][-1] if self.result is not None else None,
            trunc(self.duration) if self.duration else None,
            self.batch_size,
            self.epochs,
            self.neurons,
            self.learning_rate)

    def __repr__(self):
        return self.__str__()

    def determine_fitness(self):
        (result, seconds) = execute(self)
        self.fitness = result.history['val_loss'][-1]
        self.result = result
        self.duration = seconds
        self.info = "{id};{acc};{batch_size};{learn_rate};{epochs};{neurons};{accuracy};{val_accuracy};{loss};{val_loss};{seconds};{fitness}\n".format(
            id=self.id,
            acc=result.history['val_acc'][-1],
            batch_size=self.batch_size,
            learn_rate=self.learning_rate,
            epochs=self.epochs,
            neurons=self.neurons,
            accuracy=";".join(str(x) for x in result.history['acc']),
            val_accuracy=";".join(str(x) for x in result.history['val_acc']),
            loss=";".join(str(x) for x in result.history['loss']),
            val_loss=";".join(str(x) for x in result.history['val_loss']),
            seconds=seconds,
            fitness=self.fitness)
        print(str(self))


def store_old_generations(chromosomes):
    gen = 0
    while os.path.isfile("current" + str(gen) + ".csv"):
        gen += 1
    file = open("current" + str(gen) + ".csv", 'w')
    for gene in chromosomes:
        if gene.info is not None:
            file.write(gene.info)
    file.close()


def get_oldest_generation():
    gen = 0
    while os.path.isfile("current" + str(gen) + ".csv"):
        gen += 1
    file = open("current" + str(gen-1) + ".csv", 'r')
    tmp = int(file.readlines()[-1].split(";")[0])
    file.close()
    return


def load_generations():
    gen = 0
    if not os.path.isfile("current" + str(gen) + ".csv"):
        return True, []
    while not os.path.isfile("current" + str(gen) + ".csv"):
        gen += 1
    file = open("current" + str(gen) + ".csv", 'r')
    chromosomes = []
    for line in file:
        split = str(line).split(";")
        info = line
        chromosomes.append(Chromosome(int(split[0]), float(split[3]), int(split[2]), int(split[4]), int(split[5]), float(split[1]), info))
    file.close()
    return False, chromosomes


class Gaus:
    def __init__(self, _from, _to, variance_factor):
        self._from = _from
        self._to = _to
        self.variance_factor = variance_factor
        self.rng = np.random.default_rng()

    def compute(self, x1, x2, map = None):
        expectation = median([x1, x2])
        between = max(x1, x2) - min(x1, x2)
        sample = self.rng.normal(expectation, between/self.variance_factor, size=1)
        if map is None:
            return max(self._from, min(self._to, float(sample[0])))
        else:
            return map(max(self._from, min(self._to, float(sample[0]))))

# Tweaks
population_size = 50
elitists_count = 4
mutation_rate = 0.20

# Variables
should_store, population = load_generations()

# initiale population erzeugen
if len(population) < population_size:
    print("### Initale Population")
while len(population) < population_size:
    chromosome = Chromosome(len(population), initial_learning_rate(), initial_batch_size(), initial_epochs(), initial_neurone_size())
    print(str(chromosome))
    population.append(chromosome)

while True:
    # Fitness berechnen
    print("### Fitness Berechnen ###")
    for chromosome in population:
        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics('lineno')
        print("[ Top 10 ]")
        for stat in top_stats[:10]:
            print(stat)
        chromosome.determine_fitness()
        # if chromosome.fitness is None:

    if not should_store:
        should_store = True
    else:
        store_old_generations(population)
    # Selektion (Tournament Selection ohne zurücklegen und Elitist Selektion)
    survivors = []

    # Eltitists Selektion
    print("### Elitist")
    elitists = population[:]
    elitists.sort(key=lambda x: x.fitness)
    elitists = elitists[0:elitists_count]
    for elitist in elitists:
        print(str(elitist))
        population.remove(elitist)

    # Tournament Selektion
    def pairwise(iterable):
        a = iter(iterable)
        return zip(a, a)


    for chromosome_a, chromosome_b in pairwise(population):
        survivors.append(chromosome_a if chromosome_a.fitness <= chromosome_b.fitness else chromosome_b)

    survivors.sort(key=lambda x: x.fitness, reverse=True)
    population = survivors

    # Rekombination
    print("### Rekombination ###")
    next_id = get_oldest_generation()+1
    while len(population)+elitists_count < population_size:
        shuffle(population)
        parent1 = population[0]
        parent2 = population[1]
        id = next_id
        next_id += 1
        learning_rate = Gaus(0.00001, 0.1, 3).compute(parent1.learning_rate, parent2.learning_rate)
        batch_size = Gaus(8, 512, 3).compute(parent1.batch_size, parent2.batch_size, lambda x : int(x))
        epochs = Gaus(5, 20, 3).compute(parent1.epochs, parent2.epochs, lambda x : int(x))
        neurons = Gaus(10, 1000, 3).compute(parent1.neurons, parent2.neurons, lambda x : int(x))
        child = Chromosome(id, learning_rate, batch_size, epochs, neurons)
        print(str(chromosome))
        population.append(child)

    # Mutation
    print("### Mutation ###")
    for chromosome in population:
        # learning rate mutieren
        if uniform(0, 1) <= mutation_rate:
            chromosome.fitness = None
            chromosome.learning_rate = initial_learning_rate()
        # batch size mutieren
        if uniform(0, 1) <= mutation_rate:
            chromosome.fitness = None
            chromosome.batch_size = initial_batch_size()
        # epochs mutieren
        if uniform(0, 1) <= mutation_rate:
            chromosome.fitness = None
            chromosome.epochs = initial_epochs()
        # neurons mutieren
        if uniform(0, 1) <= mutation_rate:
            chromosome.fitness = None
            chromosome.neurons = initial_neurone_size()
        print(str(chromosome))

    for elitist in elitists:
        population.append(elitist)
