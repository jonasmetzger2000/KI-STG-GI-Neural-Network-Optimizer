import math
from random import randrange, uniform

from neural_network import execute


def initial_batch_size():
    return randrange(1, 2048)


def initial_learning_rate():
    return 10 ** uniform(math.log10(0.00001), math.log10(0.1))


class Chromosome:
    def __init__(self, id, batch_size=None, learning_rate=None):
        self.result = None
        self.id = id
        self.fitness = None
        self.info = None
        self.batch_size = initial_batch_size() if batch_size is None else batch_size
        self.learning_rate = initial_learning_rate() if learning_rate is None else learning_rate

    def __str__(self):
        return "(id={}, acc={}, fitness={}, batch_size={}, learning_rate={})".format(self.id, (self.result.history["val_acc"][9] if self.result is not None else None), self.fitness, self.batch_size, self.learning_rate)

    def __repr__(self):
        return self.__str__()

    def determine_fitness(self):
        print("Before Execution: " + str(self))
        (result, seconds) = execute(self)
        self.fitness = result.history['val_loss'][9]
        self.result = result
        self.info = "{id};{batch_size};{learn_rate:.8f};{accuracy};{val_accuracy};{loss};{val_loss};{seconds};{fitness}\n".format(
            id=self.id,
            batch_size=self.batch_size,
            learn_rate=self.learning_rate,
            accuracy=";".join(str(x) for x in result.history['acc']),
            val_accuracy=";".join(str(x) for x in result.history['val_acc']),
            loss=";".join(str(x) for x in result.history['loss']),
            val_loss=";".join(str(x) for x in result.history['val_loss']),
            seconds=seconds,
            fitness=self.fitness)
        print("After Execution: " + str(self))
