class Progress:
    def __init__(self):
        try:
            self.file = open('progress.csv', 'x+')
            self.file.write(
                "batchSize;learningRate;epoch1acc;epoch2acc;epoch3acc;epoch4acc;epoch5acc;epoch6acc;epoch7acc;" +
                "epoch8acc;epoch9acc;epoch10acc;epoch1loss;epoch2loss;epoch3loss;epoch4loss;epoch5loss;epoch6lo" +
                "ss;epoch7loss;epoch8loss;epoch9loss;epoch10loss;fitnessExecutionSeconds;fitness")
        except FileExistsError:
            self.file = open('progress.csv', 'a+')

    def write_chromosome(self, chromosome):
        self.file.write(chromosome.info)
