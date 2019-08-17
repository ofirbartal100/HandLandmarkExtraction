import matplotlib.pyplot as plt


class ProcessedImagePlotter(object):
    def __init__(self, save_flag=False):
        self.save_flag = save_flag

    def plot_single(self, processedImage, transform=None):
        # CR: doc? what is processedImage? what is transform?
        plt.clf()
        if transform:
            transformed_image = transform.transform(processedImage)
            plt.imshow(transformed_image, cmap='gray')
        else:
            plt.imshow(processedImage.img, cmap='gray')
        plt.show()

    def plot_single_grid(self, processedImage, transforms):

        if len(transforms) == 1:
            return self.plot_single(processedImage, *transforms)

        plt.clf()
        fig, axes = plt.subplots(len(transforms))

        for i, t in enumerate(transforms):
            transformed_image = t.transform(processedImage)
            axes[i].imshow(transformed_image, cmap='gray')

        if self.save_flag:
            plt.savefig("transformed_single_grid", type="jpeg")
        plt.show()

    def plot_multy_grid(self, processedImages, transforms):
        plt.clf()
        fig, axes = plt.subplots(len(transforms), len(processedImages))
        for i, t in enumerate(transforms):
            for j, pi in enumerate(processedImages):
                transformed_image = t.transform(pi)
                if len(transforms) == 1:
                    axes[j].imshow(transformed_image, cmap='gray')
                else:
                    axes[i, j].imshow(transformed_image, cmap='gray')

        if self.save_flag:
            plt.savefig("transformed_multy_grid", type="jpeg")
        plt.show()
