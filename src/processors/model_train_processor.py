from .processor import *


class ModelTrainProcessor(Processor):

    def __init__(self, trainer, **config):
        """
        Train a model with database
        :param trainer: ModelTrainer
        :param config: named vars for runing options
        """
        self.trainer = trainer
        self.config = config

    def _process(self, input):
        """
        extract landmarks from gray image tensor
        :param input: tupel (train_loader,validation_loader,test_loader) can get from database input handler
        :return: trained model
        """
        if self.config['train']:
            self.trainer.train(input[0], input[1])
        if self.config['test']:
            self.trainer.test(input[3])
        return self.trainer.get_trained_model()
