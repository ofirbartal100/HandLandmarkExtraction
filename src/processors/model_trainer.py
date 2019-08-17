import torch.nn
import logging
import numpy as np
import time
import torch
import torch.nn
import torch.nn
import torch.optim
import torch.optim
from time import gmtime, strftime


class ModelTrainer(object):
    def __init__(self, model, criterion, optimizer, to_log=False):
        self.to_log = to_log
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.temp_file_prefix = strftime("%Y-%m-%d_%H:%M_", gmtime())
        self.saved_model_file = self.temp_file_prefix + '_model.pt'
        self.device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")

        if self.to_log:
            logging.basicConfig(filename=self.temp_file_prefix + 'training.log', level='INFO')

    def train(self, train_loader, validation_loader, n_epochs=50):

        self.model.to(self.device)
        # initialize tracker for minimum validation loss
        valid_loss_min = np.Inf  # set initial "min" to infinity
        for epoch in range(n_epochs):

            self._log_print("===============================")
            self._log_print("Epoch {0}".format(epoch))
            self._log_print("===============================")

            # monitor training loss
            train_loss = 0.0
            valid_loss = 0.0

            ###################
            # train the model #
            ###################
            self.model.train()
            start_training = time.time()

            for data, target in train_loader:
                cuda_data, cuda_target = data.to(self.device), target.to(self.device)
                self.model.optimizer.zero_grad()
                output = self.model(cuda_data)
                loss = self.model.criterion(output, cuda_target)
                loss.backward()
                self.model.optimizer.step()
                train_loss += loss.item() * cuda_data.size(0)

            end_training = time.time()
            self._log_print('Finished Training in {:.6f}'.format(end_training - start_training))

            ######################
            # validate the model #
            ######################
            self.model.eval()
            start_validation = time.time()

            for data, target in validation_loader:
                cuda_data, cuda_target = data.to(self.device), target.to(self.device)
                output = self.model(cuda_data)
                loss = self.criterion(output, cuda_target)
                valid_loss += loss.item() * cuda_data.size(0)

            end_validation = time.time()
            self._log_print('Finished Validating in {:.6f}'.format(end_validation - start_validation))

            # print training/validation statistics
            # calculate average loss over an epoch
            train_loss = train_loss / len(train_loader.dataset)
            valid_loss = valid_loss / len(validation_loader.dataset)

            self._log_print(
                'Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(epoch + 1, train_loss, valid_loss))

            # save model if validation loss has decreased
            if valid_loss <= valid_loss_min:
                self._log_print(
                    'Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min,
                                                                                              valid_loss))
                torch.save(self.model.state_dict(), self.temp_file_prefix + 'model.pt')
                valid_loss_min = valid_loss

    def test(self, test_loader, specific_model_file_path):

        self._log_print("%%%%%%%%%%%%Loading Model%%%%%%%%%%%%%")

        if specific_model_file_path:
            self.model.load_state_dict(torch.load(specific_model_file_path))
        else:
            self.model.load_state_dict(torch.load(self.temp_file_prefix + 'model.pt'))

        self.model.to(self.device)

        self._log_print("%%%%%%%%Finished Loading Model%%%%%%%%")

        # initialize lists to monitor test loss and accuracy
        test_loss = 0.0
        class_correct = list(0. for i in range(10))
        class_total = list(0. for i in range(10))

        self.model.eval()  # prep model for evaluation
        start_testing = time.time()

        for data, target in test_loader:
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
        self._log_print('Finished Validating in {:.6f}'.format(end_validation - start_testing))

        # calculate and print avg test loss
        test_loss = test_loss / len(self.test_loader.dataset)
        self._log_print('Test Loss: {:.6f}\n'.format(test_loss))

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
        self.model.load_state_dict(torch.load(self.saved_model_file))
        return self.model

    def save_model(self):
        torch.save(self.model.state_dict(), self.saved_model_file)

    def _log_print(self, message):
        print(message)
        if self.to_log:
            logging.info(message)
