# import random
#
# import torchvision
# from HandsJointsDataset import *
#
# # logging.basicConfig(filename='training_resnet101.log', level=logging.INFO)
#
# images_path = r"/disk1/ofirbartal/datasets/GANeratedHands/GaneratedHands_Preprocess/" + "preprocessed_images/"
# csv_path_clean_front_hands = r"data/" + 'raw_dataframes_clean_fromt_hand.csv'
#
# in_transform = Compose([
#     CropToBoundry(),
#     Rescale((64, 64)),
#     ToTensor(),
#     Normalize((0.485,), (0.229,))
# ])
#
# # transformed_dataset = HandsJointsDataset(csv_path_clean, images_path, in_transform)
#
# # define the model
# # save_model_name = 'trained_models/resnet101_03_06_2019_model.pt'
# save_model_name = 'models/resnet50_model_only_front_hand.pt'
# # save_model_name = 'trained_models/resnet50_02_06_2019_model.pt'
# # model = torchvision.models.resnet101()
# model = torchvision.models.resnet50()
#
# model.conv1 = torch.nn.Conv2d(1, 64, (7, 7), (2, 2), (3, 3))
# model.avgpool = torch.nn.AvgPool2d(2)
# model.fc = torch.nn.Linear(2048, 42)
# #
# # criterion = torch.nn.MSELoss()
# # optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
#
# # general_model = GeneralModelRemoteServer(model, criterion, optimizer, save_model_name, transformed_dataset,to_log=True)
#
# # general_model.fit(n_epochs=20)
#
#
# def test_the_model(num_images=1):
#     directory = r"/home/ofirbartal/Projects/GaneratedHands_Preprocess/results/"
#     if not os.path.exists(directory):
#         os.makedirs(directory)
#     for n in range(num_images):
#         plt.clf()
#         test_image, test_labels = random.choice(transformed_dataset)
#         trained_model = general_model.get_trained_model()
#         trained_model.to(general_model.device)
#         start = time.time()
#         results = trained_model(test_image.unsqueeze(0).to(general_model.device)).to('cpu')
#         end = time.time()
#         print(end - start)
#         true_labels = test_labels.view(-1, 2).detach().numpy()
#         result_labels = results.view(-1, 2).detach().numpy()
#         # clip to image size
#         if (true_labels.max() > 64):
#             true_labels *= 63 / true_labels.max()
#             true_labels = np.clip(true_labels, 0, 63)
#         if (result_labels.max() > 64):
#             result_labels *= 63 / result_labels.max()
#             result_labels = np.clip(result_labels, 0, 63)
#         show_landmarks(test_image, true_labels, result_labels)
#         plt.savefig(directory+'result{0}.jpg'.format(n))
#
#
# model.load_state_dict(torch.load(self.save_model_name))
#
#
# print("cool")
# # test_the_model(100)
#
# # From Silhouette Extraction Project
# # import processedImage
# #
# # plotter = processedImage.ProcessedImagePlotter()
# #
# # image_num = [57, 100, 200, 600, 8000, 2547, 6004]
# # p_images = [processedImage.ProcessedImage(i) for i in image_num]
# #
# # transforms = [
# #     processedImage.SteeredEdgeTransform(3, 40, False),
# #     processedImage.CannyTransform(100, 200, True)
# # ]
#
# # plotter.plot_multy_grid(p_images, transforms)
