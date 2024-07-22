from chromosome import Chromosome
from neural_network import NeuralNetwork

neural_network = NeuralNetwork()
f1 = Chromosome(1, neural_network, 1732, 1.0044068105651245e-05)
f1.determine_fitness()
print(f1)

