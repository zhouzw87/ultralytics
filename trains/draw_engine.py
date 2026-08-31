import torch
import torchvision.models as models
from trex import EnginePlan
from trex import to_dot
from trex import layer_type_formatter
from trex import precision_formatter
from trex import render_dot
import os
from pytorch_quantization import quant_modules  # For QAT
from pytorch_quantization import nn as quant_nn


def convert_onnx():
    quant_modules.initialize()
    quant_nn.TensorQuantizer.use_fb_fake_quant = True

    resnet = models.resnet18(pretrained=True).eval()

    with torch.no_grad():  # Export to ONNX, with dynamic batch-size
        input = torch.randn(1, 3, 224, 224)
        torch.onnx.export(resnet,
                          input,
                          "resnet-qat.onnx",
                          input_names=["input.1"],
                          opset_version=13,
                          dynamic_axes={"input.1": {0: "batch_size"}})


def generate_json_info():
    os.system(
        # "python3 /home/admin1/TensorRT/tools/experimental/trt-engine-explorer/utils/process_engine.py yolov8m.onnx yolov8m_int8 int8 fp16 calib=yolov8_calib.cache shapes=images:1x3x640x640")
        # "python3 /home/admin1/TensorRT/tools/experimental/trt-engine-explorer/utils/process_engine.py yolov8m.onnx yolov8m fp16 shapes=images:1x3x640x640")
        "python3 /home/admin1/TensorRT/tools/experimental/trt-engine-explorer/utils/process_engine.py qat.onnx qat int8 fp16 shapes=images:1x3x640x640")
        # "python3 /home/admin1/TensorRT/tools/experimental/trt-engine-explorer/utils/process_engine.py modified.onnx modified int8 fp16 shapes=images:1x3x640x640")

#python3 process_engine.py qat.onnx qat int8 fp16 shapes=images:1x3x640x640
def generate_svg():
    engine_name = "qat/qat.onnx.engine"
    plan = EnginePlan(f"{engine_name}.graph.json",
                      f"{engine_name}.profile.json",
                      f"{engine_name}.profile.metadata.json")
    graph = to_dot(plan, layer_type_formatter)
    svg_name = render_dot(graph, engine_name, 'svg')


if __name__ == "__main__":
    # convert_onnx()
    generate_json_info()
    generate_svg()
    #LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64
