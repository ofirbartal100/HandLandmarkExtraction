import logging
import re
import cv2
import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torchvision import transforms, utils
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TTF
import time
import torch.nn
import torch.optim


class HandsJointsDataset(Dataset):
    """Tracker Position dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.joints_pos_dataframes = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.joints_pos_dataframes)

    def __getitem__(self, idx):
        record = self.joints_pos_dataframes.iloc[idx]
        image_name = record[0]
        # print("{0}:{1}".format(idx,image_name))
        image_path = os.path.join(self.root_dir, image_name)

        joints_pos = np.array(record[1:], dtype='float')
        joints_pos = joints_pos.reshape(-1, 2)

        if self.transform:
            pil_image = Image.open(image_path).convert('L')
            return self.transform(pil_image, joints_pos)

        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        in_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485,),
                                 (0.229,))])

        # discard the transparent, alpha channel (that's the :3) and add the batch dimension
        image = in_transform(image)[:, :, :]

        return image, joints_pos


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, pil_image, joints_pos):
        w, h = pil_image.size
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        rescaled_image = TTF.resize(pil_image, self.output_size)

        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively
        joints_pos = joints_pos * [new_w / w, new_h / h]
        return rescaled_image, joints_pos


class CropToBoundry(object):
    def __call__(self, pil_image, joints_pos):
        w, h = pil_image.size
        max_xy = np.max(joints_pos, 0) + 10  # padding
        min_xy = np.min(joints_pos, 0) - 10  # padding

        max_xy = np.clip(max_xy, 0, [w, h])
        min_xy = np.clip(min_xy, 0, [w, h])
        new_wh = max_xy - min_xy
        cropped_image = TTF.crop(pil_image, min_xy[1], min_xy[0], new_wh[1], new_wh[0])

        cropped_joints_pos = joints_pos - min_xy
        return cropped_image, cropped_joints_pos


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, pil_image, joints_pos):
        for transform in self.transforms:
            pil_image, joints_pos = transform(pil_image, joints_pos)

        return pil_image, joints_pos


class ToTensor(object):
    def __init__(self):
        pass

    def __call__(self, pil_image, joints_pos):
        return TTF.to_tensor(pil_image), torch.from_numpy(joints_pos).type(torch.FloatTensor).view(-1)


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, pil_image_tensor, joints_pos):
        return TTF.normalize(pil_image_tensor, self.mean, self.std), joints_pos


class GeneralModel:
    def __init__(self, model, criterion, optimizer, save_model_name, transformed_dataset, batch_size=64, to_log=False):
        self.to_log = to_log
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.save_model_name = save_model_name
        if self.to_log:
            logging.basicConfig(filename=save_model_name + 'training.log', level='INFO')
        # create samplers for the loaders
        train_sampler, valid_sampler, test_sampler = data_samplers(transformed_dataset, valid_size=0.2, test_size=0.2)
        self.train_loader = DataLoader(transformed_dataset, batch_size=batch_size, sampler=train_sampler)
        self.valid_loader = DataLoader(transformed_dataset, batch_size=batch_size, sampler=valid_sampler)
        self.test_loader = DataLoader(transformed_dataset, batch_size=batch_size, sampler=test_sampler)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit(self, n_epochs=50, to_print=True):
        # def nothing(string):5
        #     pass
        #
        # new_print = print if to_print else nothing

        self.model.to(self.device)
        # initialize tracker for minimum validation loss
        valid_loss_min = np.Inf  # set initial "min" to infinity
        for epoch in range(n_epochs):

            print("===============================")
            print("Epoch {0}".format(epoch))
            print("===============================")

            if self.to_log:
                logging.info("===============================")
                logging.info("Epoch {0}".format(epoch))
                logging.info("===============================")

            # monitor training loss
            train_loss = 0.0
            valid_loss = 0.0

            ###################
            # train the model #
            ###################
            self.model.train()
            start_training = time.time()

            for data, target in self.train_loader:
                cuda_data, cuda_target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(cuda_data)
                loss = self.criterion(output, cuda_target)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item() * cuda_data.size(0)

            end_training = time.time()
            # print('Finished Training in {:.6f}'.format(end_training - start_training))
            print('Finished Training in {:.6f}'.format(end_training - start_training))
            if self.to_log:
                logging.info('Finished Training in {:.6f}'.format(end_training - start_training))

            ######################
            # validate the model #
            ######################
            self.model.eval()
            start_validation = time.time()

            for data, target in self.valid_loader:
                cuda_data, cuda_target = data.to(self.device), target.to(self.device)
                output = self.model(cuda_data)
                loss = self.criterion(output, cuda_target)
                valid_loss += loss.item() * cuda_data.size(0)

            end_validation = time.time()
            # print('Finished Validating in {:.6f}'.format(end_validation - start_validation))
            print('Finished Validating in {:.6f}'.format(end_validation - start_validation))
            if self.to_log:
                logging.info('Finished Validating in {:.6f}'.format(end_validation - start_validation))

            # print training/validation statistics
            # calculate average loss over an epoch
            train_loss = train_loss / len(self.train_loader.dataset)
            valid_loss = valid_loss / len(self.valid_loader.dataset)

            # print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(epoch + 1, train_loss, valid_loss))
            print(
                'Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(epoch + 1, train_loss, valid_loss))
            if self.to_log:
                logging.info('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(epoch + 1, train_loss,
                                                                                                  valid_loss))

            # save model if validation loss has decreased
            if valid_loss <= valid_loss_min:
                print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min,
                                                                                                valid_loss))
                if self.to_log:
                    logging.info(
                        'Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min,
                                                                                                  valid_loss))
                # print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min, valid_loss))
                torch.save(self.model.state_dict(), self.save_model_name)
                valid_loss_min = valid_loss

    def test(self, to_print=True):
        # def nothing(string):
        #     pass
        #
        # new_print = print if to_print else nothing

        print("%%%%%%%%%%%%Loading Model%%%%%%%%%%%%%")
        if self.to_log:
            logging.info("%%%%%%%%%%%%Loading Model%%%%%%%%%%%%%")

        self.model.load_state_dict(torch.load(self.save_model_name))
        self.model.to(self.device)
        print("%%%%%%%%Finished Loading Model%%%%%%%%")
        if self.to_log:
            logging.info("%%%%%%%%Finished Loading Model%%%%%%%%")

        # initialize lists to monitor test loss and accuracy
        test_loss = 0.0
        class_correct = list(0. for i in range(10))
        class_total = list(0. for i in range(10))

        self.model.eval()  # prep model for evaluation
        start_testing = time.time()

        for data, target in self.test_loader:
            cuda_data, cuda_target = data.to(self.device), target.to(self.device)
            output = self.model(cuda_data)
            loss = self.criterion(output, cuda_target)
            test_loss += loss.item() * data.size(0)
            # # convert output probabilities to predicted class
            # _, pred = torch.max(output, 1)
            # # compare predictions to true label
            # correct = np.squeeze(pred.eq(target.data.view_as(pred)))
            # # calculate test accuracy for each object class
            # for i in range(len(target)):
            #     label = target.data[i]
            #     class_correct[label] += correct[i].item()
            #     class_total[label] += 1

        end_validation = time.time()
        # print('Finished Validating in {:.6f}'.format(end_validation - start_validation))
        print('Finished Validating in {:.6f}'.format(end_validation - start_testing))
        if self.to_log:
            logging.info('Finished Validating in {:.6f}'.format(end_validation - start_testing))

        # calculate and print avg test loss
        test_loss = test_loss / len(self.test_loader.dataset)
        print('Test Loss: {:.6f}\n'.format(test_loss))
        if self.to_log:
            logging.info('Test Loss: {:.6f}\n'.format(test_loss))

        # for i in range(10):
        #     if class_total[i] > 0:
        #         print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
        #             str(i), 100 * class_correct[i] / class_total[i],
        #             np.sum(class_correct[i]), np.sum(class_total[i])))
        #     else:
        #         print('Test Accuracy of %5s: N/A (no training examples)' % (classes[i]))
        #
        # print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
        #     100. * np.sum(class_correct) / np.sum(class_total),
        #     np.sum(class_correct), np.sum(class_total)))

    def get_trained_model(self):
        self.model.load_state_dict(torch.load(self.save_model_name))
        return self.model


class GeneralModelRemoteServer(GeneralModel):
    def __init__(self, model, criterion, optimizer, save_model_name, transformed_dataset, batch_size=64, to_log=False):
        super(GeneralModelRemoteServer, self).__init__(model, criterion, optimizer, save_model_name,
                                                       transformed_dataset, batch_size=batch_size, to_log=to_log)
        self.device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")


def data_samplers(transformed_dataset, valid_size=0.2, test_size=0.2):
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


def raw_dataset_to_dataframes(input_Images_folders_path, output_preprocessed_path):
    if not os.path.exists(output_preprocessed_path):
        os.makedirs(output_preprocessed_path)

    output_preprocessed_images_folder_path = output_preprocessed_path + "preprocessed_images/"
    if not os.path.exists(output_preprocessed_images_folder_path):
        os.makedirs(output_preprocessed_images_folder_path)

    dataframes_file = open(output_preprocessed_path + 'raw_dataframes.csv', 'w')
    dataframes_file.write(
        'image_name,Wx,Wy,T0x,T0y,T1x,T1y,T2x,T2y,T3x,T3y,I0x,I0y,I1x,I1y,I2x,I2y,I3x,I3y,M0x,M0y,M1x,M1y,M2x,M2y,M3x,M3y,R0x,R0y,R1x,R1y,R2x,R2y,R3x,R3y,L0x,L0y,L1x,L1y,L2x,L2y,L3x,L3y\n')

    media_extentions = ['.jpg', '.png', 'jpeg']

    count_folders = len(os.listdir(input_Images_folders_path))
    counter = 0
    for folder_name in os.listdir(input_Images_folders_path):
        counter += 1
        print("processing #{0} out of {1}".format(counter, count_folders))
        if not os.path.isdir(input_Images_folders_path + folder_name):
            continue
        current_folder_path = input_Images_folders_path + folder_name
        images_file_names = [fn for fn in os.listdir(current_folder_path)
                             if any(fn.endswith(ext) for ext in media_extentions)]
        for image_file_name in images_file_names:
            image_prefix = re.search("(\d{4})_", image_file_name).group(1)
            with open(
                    current_folder_path + "/" + image_prefix + "_joint2D.txt") as joints_file:  # Use file to refer to the file object
                data = joints_file.read()
                new_image_name = "{0}_{1}".format(folder_name, image_file_name)
                dataframe = "{0},{1}".format(new_image_name, data)
                dataframes_file.write(dataframe)

                # Load an color image in grayscale
                img = cv2.imread(current_folder_path + "/" + image_file_name, cv2.IMREAD_GRAYSCALE)
                cv2.imwrite(output_preprocessed_images_folder_path + new_image_name, img)


import matplotlib.pyplot as plt


def draw_skeleton(landmarks, style):
    # draw segments from wrist
    for i in [1, 5, 9, 13, 17]:
        plt.plot([landmarks[0, 0], landmarks[i, 0]], [landmarks[0, 1], landmarks[i, 1]], style)

    # draw fingers skeleton
    for f in [1, 5, 9, 13, 17]:
        for j in range(3):
            plt.plot([landmarks[f + j, 0], landmarks[f + j + 1, 0]], [landmarks[f + j, 1], landmarks[f + j + 1, 1]],
                     style)


def show_landmarks(image, true_lanndmarks, results_landmarks):
    """Show image with landmarks"""
    im = np.transpose(image.numpy(), (1, 2, 0))
    sq = im.squeeze()
    plt.imshow(sq, cmap='gray')
    draw_skeleton(true_lanndmarks, 'ro-')
    draw_skeleton(results_landmarks, 'bo-')


ganerates_dataset_path = r"/disk1/ofirbartal/datasets/GANeratedHands/GaneratedDataset_extracted/GANeratedHands_Release/data/noObject/"
output_path = r"/disk1/ofirbartal/datasets/GANeratedHands/GaneratedHands_Preprocess/"
images_path = r"/disk1/ofirbartal/datasets/GANeratedHands/GaneratedHands_Preprocess/" + "preprocessed_images/"
csv_path_clean = r"/disk1/ofirbartal/datasets/GANeratedHands/GaneratedHands_Preprocess/" + 'raw_dataframes_clean.csv'
