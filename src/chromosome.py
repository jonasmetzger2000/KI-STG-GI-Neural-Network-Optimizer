import math
from random import randrange, uniform

from neural_network import execute


def initial_batch_size():
    return randrange(1, 2048)


def initial_learning_rate():
    return 10 ** uniform(math.log10(0.00001), math.log10(0.1))


def initial_epochs():
    return randrange(5, 30)


class Chromosome:
    def __init__(self, id, batch_size=None, learning_rate=None, epochs=None, fitness=None, info=None):
        self.result = None
        self.id = id
        self.info = info
        self.fitness = fitness
        self.batch_size = initial_batch_size() if batch_size is None else batch_size
        self.learning_rate = initial_learning_rate() if learning_rate is None else learning_rate
        self.epochs = initial_epochs() if epochs is None else epochs

    def __str__(self):
        return "(id={}, acc={}, fitness={}, batch_size={}, learning_rate={}, epochs={})".format(self.id, (self.result.history["val_acc"][self.epochs-1] if self.result is not None else None), self.fitness, self.batch_size, self.learning_rate, self.epochs)

    def __repr__(self):
        return self.__str__()

    def determine_fitness(self):
        print("Before Execution: " + str(self))
        (result, seconds) = execute(self)
        self.fitness = result.history['val_loss'][self.epochs-1]
        self.result = result
        self.info = "{id};{batch_size};{learn_rate};{epochs};{accuracy};{val_accuracy};{loss};{val_loss};{seconds};{fitness}\n".format(
            id=self.id,
            batch_size=self.batch_size,
            learn_rate=self.learning_rate,
            epochs=self.epochs,
            accuracy=";".join(str(x) for x in result.history['acc']),
            val_accuracy=";".join(str(x) for x in result.history['val_acc']),
            loss=";".join(str(x) for x in result.history['loss']),
            val_loss=";".join(str(x) for x in result.history['val_loss']),
            seconds=seconds,
            fitness=self.fitness)
        print("After Execution: " + str(self))
