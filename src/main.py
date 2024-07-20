from chromosome import Chromosome
from neural_network import NeuralNetwork
from progress import Progress

progress = Progress()

neural_network = NeuralNetwork()
f1 = Chromosome(neural_network)
f1.determine_fitness()
progress.write_chromosome(f1)

