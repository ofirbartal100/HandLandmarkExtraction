from .processor import *


class ExtractLandmarksProcessor(Processor):

    def __init__(self, model):
        self.model = model

    def _process(self, input):
        """
        extract landmarks from gray image tensor
        :param input: gray image tensor
        :return: 21 landmarks array
        """
        return (input, self.model(input.unsqueeze(0)).view(-1, 2))
