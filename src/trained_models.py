import torchvision
from HandsJointsDataset import *

images_path = r"/disk1/ofirbartal/datasets/GANeratedHands/GaneratedHands_Preprocess/" + "preprocessed_images/"
csv_path_clean = r"/disk1/ofirbartal/datasets/GANeratedHands/GaneratedHands_Preprocess/" + 'raw_dataframes_clean.csv'

in_transform = Compose([
    CropToBoundry(),
    Rescale((64, 64)),
    ToTensor(),
    Normalize((0.485,), (0.229,))
])

transformed_dataset = HandsJointsDataset(csv_path_clean, images_path, in_transform)

# define the model
# save_model_name = 'trained_models/resnet101_model.pt'
save_model_name = 'trained_models/resnet50_02_06_2019_model.pt'

# model = torchvision.models.resnet101()
model = torchvision.models.resnet50()

model.conv1 = torch.nn.Conv2d(1, 64, (7, 7), (2, 2), (3, 3))
model.avgpool = torch.nn.AvgPool2d(2)
model.fc = torch.nn.Linear(2048, 42)

criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

general_model = GeneralModelRemoteServer(model, criterion, optimizer, save_model_name, transformed_dataset)