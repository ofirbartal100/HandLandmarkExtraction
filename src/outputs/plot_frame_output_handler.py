from .output_handler import OutputHandler
import matplotlib.pyplot as plt


class PlotFrameOutputHandler(OutputHandler):
    def __init__(self, file_name=False):
        self._file_name = file_name

    def _handle(self, *args):
        image, landmarks = args[0]
        plt.clf()
        plt.imshow(image.numpy().squeeze(), cmap='gray')
        self._show_landmarks(landmarks.detach().numpy())

        if self._file_name:
            plt.savefig(self._file_name)
        plt.show()

    def _show_landmarks(self, landmarks):
        for landmark in landmarks:
            plt.scatter(*landmark, c='r')
