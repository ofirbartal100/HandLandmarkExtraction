import os
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.utils.data import Dataset
from .input_handler import *
from .dataset_transforms import *


class GaneratedHandsDataset(Dataset):
    """Ganerated Hands's images labeled with joints lanmarks dataset."""

    def __init__(self, transform, labels_csv_path, images_root_directory_path):
        """
        Args:
            labels_csv_path (string): Path to the csv file with annotations.
            images_root_directory_path (string): Directory with all the images.
        """
        self.transform = transform
        self.labels = pd.read_csv(labels_csv_path)
        self.root_dir = images_root_directory_path

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        """
        query data from the csv file
        :param idx: row index in csv file
        :return: PIL grayscale image , 21 landmarks label
        """
        record = self.labels.iloc[idx]

        # get image name + images root directory
        image_path = os.path.join(self.root_dir, record[0])
        image = Image.open(image_path).convert('L')

        label = np.array(record[1:], dtype='float')
        label = label.reshape(-1, 2)

        return self.transform(image, label)


class GaneratedHandsDatabase():

    def __init__(self, transform, labels_csv_path, images_root_directory_path, batch_size=64):
        self.dataset = GaneratedHandsDataset(transform, labels_csv_path, images_root_directory_path)

        # create samplers for the loaders
        train_sampler, valid_sampler, test_sampler = self._data_samplers(self.dataset)
        self.train_loader = DataLoader(self.dataset, batch_size=batch_size, sampler=train_sampler)
        self.valid_loader = DataLoader(self.dataset, batch_size=batch_size, sampler=valid_sampler)
        self.test_loader = DataLoader(self.dataset, batch_size=batch_size, sampler=test_sampler)

    def _data_samplers(self, transformed_dataset, valid_size=0.2, test_size=0.2):
        # obtain training indices that will be used for validation
        num_train = len(transformed_dataset)
        indices = list(range(num_train))
        np.random.shuffle(indices)
        split = int(np.floor(test_size * num_train))
        train_valid__idx, test_idx = indices[split:], indices[:split]
        # define samplers for obtaining training and validation batches
        test_sampler = SubsetRandomSampler(test_idx)

        num_train2 = len(train_valid__idx)
        split2 = int(np.floor(valid_size * num_train2))
        train_idx, valid_idx = train_valid__idx[split2:], train_valid__idx[:split2]
        # define samplers for obtaining training and validation batches
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)
        return train_sampler, valid_sampler, test_sampler


class DataBaseInputHandler(InputHandler):

    def _handle(self, *args):
        """
        return the dataset loaders
        :param args: labels_csv_path, images_root_directory_path, *batch_size = 64
        """

        # this transform was used during the first training, we crop image to boundry of hand,
        # than rescale it to 64x64, than transform to tensor, and finally normalize
        transform = Compose([
            CropToBoundry(),
            Rescale((64, 64)),
            ToTensor(),
            Normalize((0.485,), (0.229,))
        ])

        db = GaneratedHandsDatabase(transform, *args)
        return (db.train_loader, db.valid_loader, db.test_loader)
