from random import choice


def initial_batch_size():
    return choice([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048])


def initial_learning_rate():
    return choice([1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1])


generation = 0


class Chromosome:
    def __init__(self, neural_network):
        global generation
        self.generation = generation
        generation += 1
        self.fitness = None
        self.info = None
        self.batch_size = initial_batch_size()
        self.learning_rate = initial_learning_rate()
        self.neural_network = neural_network

    def determine_fitness(self):
        (result, seconds) = self.neural_network.execute(self)
        self.fitness = result.history['loss'][9]
        self.info = "{generation};{batch_size};{learn_rate:.8f};{accuracy};{loss};{seconds};{fitness}\n".format(
            generation=self.generation,
            batch_size=self.batch_size,
            learn_rate=self.learning_rate,
            accuracy=";".join(str(x) for x in result.history['acc']),
            loss=";".join(str(x) for x in result.history['loss']),
            seconds=seconds,
            fitness=self.fitness)
