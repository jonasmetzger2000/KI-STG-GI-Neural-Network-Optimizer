import math
from random import randrange, uniform
from math import trunc

from neural_network import execute


def initial_batch_size():
    return randrange(8, 512)


def initial_learning_rate():
    return 10 ** uniform(math.log10(0.00001), math.log10(0.1))


def initial_epochs():
    return randrange(5, 20)


def initial_neurone_size():
    return randrange(10, 1000)


class Chromosome:
    def __init__(self, id, learning_rate, batch_size, epochs, neurons, fitness = None, info = None):
        self.result = None
        self.fitness = None
        self.duration = None
        self.info = info
        self.id = id
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.neurons = neurons
        self.fitness = fitness

    def __str__(self):
        return ("(id={}, acc={}, fitness={}, batch_size={}, learning_rate={}, epochs={}, neurons={}, seconds={})"
                .format(self.id,
                        self.result.history["val_acc"][self.epochs-1] if self.result is not None else None,
                        self.fitness,
                        self.batch_size,
                        self.learning_rate,
                        self.epochs,
                        self.neurons,
                        trunc(self.duration) if self.duration else None))

    def __repr__(self):
        return self.__str__()

    def determine_fitness(self):
        print("Before Execution: " + str(self))
        (result, seconds) = execute(self)
        self.fitness = result.history['val_loss'][self.epochs-1]
        self.result = result
        self.duration = seconds
        self.info = "{id};{acc}{batch_size};{learn_rate};{epochs};{neurons};{accuracy};{val_accuracy};{loss};{val_loss};{seconds};{fitness}\n".format(
            id=self.id,
            acc=result.history['val_acc'][self.epochs-1],
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
        print("After Execution: " + str(self))
