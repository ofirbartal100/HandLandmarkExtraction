from inputs.database_input_handler import DataBaseInputHandler
from inputs.dataset_transforms import *
from processors.model_train_processor import ModelTrainProcessor
from processors.model_trainer import ModelTrainer
from models.landmarks_extraction_model import LandmarksExtractionModel

from inputs.image_input_handler import PILGrayImageInputHandler,TensorImageInputHandler
from torchvision import transforms
from inputs.video_frame_input_handler import TensorVideoFrameInputHandler,PILGrayVideoFrameInputHandler
from processors.extract_landmarks_processor import ExtractLandmarksProcessor
from outputs.plot_frame_output_handler import PlotFrameOutputHandler
from models.landmarks_extraction_model import LandmarksExtractionModel
from PIL import Image

# LandmarksExtraction From VideoFrame
# input_handler = TensorVideoFrameInputHandler()
# input = input_handler.handle("/disk1/ofirbartal/Projects/LandmarksExtraction/Videos/IR_Hands_Video_01.avi", 3000,
#                              (64, 64))
#


# LandmarksExtraction From File
# input_handler = PILGrayVideoFrameInputHandler()
# input = input_handler.handle("/disk1/ofirbartal/Projects/LandmarksExtraction/Videos/IR_Hands_Video_01.avi", 500)
# input.save("temp_frame2.jpg")
# exit()
#
# input_handler = PILGrayImageInputHandler()
# pil = input_handler.handle("temp_frame2.jpg")
# # bb1= (264,155,467,475)
# bb2 = (220,208,492,469)
# bb_pil = pil.crop(bb2)
# resized_pil = bb_pil.resize([64,64])
# flipped = resized_pil.transpose(Image.FLIP_LEFT_RIGHT)
# flipped.save("transformed_temp_frame_flipped2.jpg")
# exit()

input_handler = TensorImageInputHandler()
input = input_handler.handle("transformed_temp_frame_flipped2.jpg")
model = LandmarksExtractionModel("models/resnet50_model_only_front_hand.pt")
processor = ExtractLandmarksProcessor(model)
output = processor.process(input)

output_handler = PlotFrameOutputHandler("temp_trensformed2.png")
output_handler.handle(output)


# ModelTraining
#
# input_handler = DataBaseInputHandler()
# input = input_handler.handle("Data/temp.csv",
#                              "/disk1/ofirbartal/Projects/LandmarksExtraction/Dataset/")
#
# model = LandmarksExtractionModel("models/resnet50_model_only_front_hand.pt")
# trainer = ModelTrainer(model, torch.nn.MSELoss(), torch.optim.SGD(model.parameters(), lr=0.01), True)
# processor = ModelTrainProcessor(trainer, train=True, test=True)
#
# output = processor.process(input)
# torch.save(output.state_dict(), "models/resnet50_250x250_front.pt")


