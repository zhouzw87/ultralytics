from ultralytics import YOLO

# Load a model
# model = YOLO("../runs/detect/train/weights/last.pt")  # load a pretrained model (recommended for training)
# success = model.export(format="onnx", opset=13, simplify=False)  # export the model to onnx format
# assert success

model = YOLO("../runs/detect/train_peoplecar_yolov8m/weights/last.pt")  # load a pretrained model (recommended for training)
success = model.export(format="onnx", opset=13, simplify=False,dynamic=False)  # export the model to onnx format
assert success