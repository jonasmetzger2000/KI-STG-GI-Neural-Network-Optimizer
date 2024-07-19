from random import choice


def initial_batch_size():
    return choice([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048])


def initial_learning_rate():
    return choice([1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1])



class Chromosome:
    def __init__(self):
        self.batch_size = initial_batch_size()
        self.learning_rate = initial_learning_rate()
