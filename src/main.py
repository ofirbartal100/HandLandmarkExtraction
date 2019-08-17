from .inputs.database_input_handler import DataBaseInputHandler
from .inputs.dataset_transforms import *
from .processors.model_train_processor import ModelTrainProcessor
from .processors.model_trainer import ModelTrainer
from .models.landmarks_extraction_model import LandmarksExtractionModel

# from inputs.video_frame_input_handler import TensorVideoFrameInputHandler
# from processors.extract_landmarks_processor import ExtractLandmarksProcessor
# from outputs.plot_frame_output_handler import PlotFrameOutputHandler
# from models.landmarks_extraction_model import LandmarksExtractionModel


# LandmarksExtraction From VideoFrame
# input_handler = TensorVideoFrameInputHandler()
# input = input_handler.handle("/disk1/ofirbartal/Projects/LandmarksExtraction/Videos/IR_Hands_Video_01.avi", 3000,
#                              (64, 64))
#
# model = LandmarksExtractionModel("models/resnet50_model_only_front_hand.pt")
# processor = ExtractLandmarksProcessor(model)
# output = processor.process(input)
#
# output_handler = PlotFrameOutputHandler("temp6.png")
# output_handler.handle(output)


# ModelTraining

input_handler = DataBaseInputHandler()
input = input_handler.handle("Data/temp.csv",
                             "/disk1/ofirbartal/Projects/LandmarksExtraction/Dataset/")

model = LandmarksExtractionModel("models/resnet50_model_only_front_hand.pt")
trainer = ModelTrainer(model, torch.nn.MSELoss(), torch.optim.SGD(model.parameters(), lr=0.01), True)
processor = ModelTrainProcessor(trainer, train=True, test=True)

output = processor.process(input)
torch.save(output.state_dict(), "models/resnet50_250x250_front.pt")
