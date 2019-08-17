from inputs.video_frame_input_handler import TensorVideoFrameInputHandler
from processors.extract_landmarks_processor import ExtractLandmarksProcessor
from outputs.plot_frame_output_handler import PlotFrameOutputHandler
from models.landmarks_extraction_model import LandmarksExtractionModel

input_handler = TensorVideoFrameInputHandler()
input = input_handler.handle("/disk1/ofirbartal/Projects/LandmarksExtraction/Videos/IR_Hands_Video_01.avi", 3000,
                             (64, 64))

model = LandmarksExtractionModel("models/resnet50_model_only_front_hand.pt")
processor = ExtractLandmarksProcessor(model)
output = processor.process(input)

output_handler = PlotFrameOutputHandler("temp6.png")
output_handler.handle(output)
