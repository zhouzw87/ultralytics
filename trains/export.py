"""ONNX export for YOLO detection checkpoints, one output layout per deployment target.

Independent of the QAT code: pytorch_quantization is only needed to export a quantized
checkpoint, so a plain FP32 export works without it installed.
"""

import os

import onnx
import onnx_graphsurgeon as gs
import onnxsim
import torch

from ultralytics.nn.modules import Detect
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import LOGGER, colorstr
from ultralytics.utils.checks import check_requirements

try:  # only a QAT checkpoint needs fake-quant nodes in the traced graph
    import quantize
except ImportError:
    quantize = None


def torch2onnx(model, input, file, **kwargs):
    """Trace to ONNX, emitting fake-quant nodes when pytorch_quantization is in play."""
    if quantize is not None:
        quantize.quant_nn.TensorQuantizer.use_fb_fake_quant = True
    try:
        with torch.no_grad():
            torch.onnx.export(model, input, file, **kwargs)
    finally:
        if quantize is not None:
            quantize.quant_nn.TensorQuantizer.use_fb_fake_quant = False


OUTPUT_NAMES = {  # each output layout is mutually exclusive and decides the ONNX output names
    'raw': ['output0'],
    'transpose': ['output0'],
    'end2end': ['num_dets', 'det_boxes', 'det_scores', 'det_classes'],
    'split': ['bbox', 'conf', 'class_id'],
    'rknn': [],  # one reg/cls pair per detection scale, derived from the head below
}


def export_onnx(model: DetectionModel, save_file, size=640, dynamic_batch=False,
                output='raw', simplify=False, ort=False, prefix=colorstr('ONNX:')):
    requirements = ['onnx>=1.12.0']
    check_requirements(requirements)

    device = next(model.parameters()).device
    detects = [m for m in model.modules() if isinstance(m, Detect)]

    if output in {'end2end', 'transpose', 'split'} and any(m.end2end for m in detects):
        raise ValueError(f"output='{output}' needs the raw (batch, 4 + nc, anchors) head output, but this checkpoint "
                         f"is end-to-end and already returns (batch, max_det, 6). Use output='raw' or 'rknn'.")

    flags = [(m.export, m.format, m.dynamic) for m in detects]
    training = model.training
    for m in detects:  # trace in export mode, otherwise Detect also returns its internal preds
        m.export, m.format, m.dynamic = True, 'onnx', dynamic_batch
    model.eval()  # QAT checkpoints are saved mid-finetune, so they come back in train mode

    batch_size = 1
    input = torch.zeros(batch_size, 3, size, size).to(device)

    output_names = OUTPUT_NAMES[output] or [f'{k}{i + 1}' for i in range(detects[0].nl) for k in ('reg', 'cls')]
    dynamic_axes = None
    if dynamic_batch:
        batch_size = 'batch'
        dynamic_axes = {'images': {0: 'batch', 2: "height", 3: "width"}}
        dynamic_axes.update({name: {0: 'batch'} for name in output_names})

    if output == 'end2end':
        from end2end import End2End
        model = End2End(model, max_obj=100, iou_thres=0.45,score_thres=0.5,
                        device=device, ort=ort, trt_version=8, with_preprocess=False)
    elif output == 'transpose':
        from end2end import TransOut
        model = TransOut(model, device=device)
    elif output == 'split':
        from end2end import SplitOut
        model = SplitOut(model, device=device)
    elif output == 'rknn':
        from ultralytics.utils.export.rknn import rknn_wrapper
        model = rknn_wrapper(model)
    if dynamic_batch:  # dynamic=True only compatible with cpu
        model, input = model.cpu(), input.cpu()
    with torch.no_grad():  # warm up so make_anchors runs outside the trace, which keeps the anchors fp32
        model(input)
    torch2onnx(model, input, save_file, verbose=False, opset_version=13, do_constant_folding=True,
               input_names=['images'], output_names=output_names, dynamic_axes=dynamic_axes)
    for m, flag in zip(detects, flags):  # run_qat exports mid-finetune, so leave the live model as we found it
        m.export, m.format, m.dynamic = flag
    model.train(training)
    # Simplify
    onnx_model = onnx.load(save_file)  # load onnx model
    onnx.checker.check_model(onnx_model)  # check onnx model
    # Fix output shape
    if output == 'end2end' and not ort:
        topk_all=100
        shapes = [batch_size, 1, batch_size, topk_all, 4,
                  batch_size, topk_all, batch_size, topk_all]
        for i in onnx_model.graph.output:
            for j in i.type.tensor_type.shape.dim:
                j.dim_param = str(shapes.pop(0))
    if simplify:
        try:
            LOGGER.info(f'{prefix} simplifying with onnxsim {onnxsim.__version__}...')
            # subprocess.run(f'onnxsim "{f}" "{f}"', shell=True)
            onnx_model, check = onnxsim.simplify(onnx_model)
            assert check, 'Simplified ONNX model could not be validated'
        except Exception as e:
            LOGGER.info(f'{prefix} simplifier failure: {e}')
    onnx.save(onnx_model, save_file)
    print(f"Save onnx to {save_file}")


def graphsurgeon_model(save_file):
    """Rewrite a QAT graph in place so TensorRT can fuse it, sharing Q/DQ across an activation's branches."""
    graph = gs.import_onnx(onnx.load(save_file))
    if not any(node.op == "DequantizeLinear" for node in graph.nodes):
        raise ValueError("--graphsurgeon expects a QAT graph with Q/DQ nodes, but this export has none")

    # A SiLU Mul feeding several consumers gets one QuantizeLinear per branch, each with its own scale,
    # which TensorRT cannot fuse. Rewire the extra branches onto the first branch's DQ so they share it.
    for node in graph.nodes:
        if node.op != "Mul":
            continue
        consumers = node.outputs[0].outputs
        if len(consumers) < 2:
            continue
        shared_dq = node.o(0).o(0).outputs[0]  # output of the first branch's DequantizeLinear

        if len(consumers) == 2 and consumers[0].op == "QuantizeLinear":
            if consumers[1].op == "QuantizeLinear":
                target = node.o(1).o(0).o(0)  # consumer past the second branch's Q/DQ pair
                replaced = node.o(1).o(0).outputs[0].name
            elif consumers[1].op == "Concat":
                target = consumers[1]
                replaced = consumers[0].inputs[0].name
            else:
                continue
            if target.op == "Concat":  # a Concat takes the branch at whichever input index it sits on
                for i, inp in enumerate(target.inputs):
                    if inp.name == replaced:
                        target.inputs[i] = shared_dq
            else:
                target.inputs[0] = shared_dq
        elif len(consumers) in {3, 4}:  # with 4, branch 1 is an unmerged Shape node and is left alone
            for branch in ([2, 1] if len(consumers) == 3 else [3, 2]):
                node.o(branch).o(0).o(0).inputs[0] = shared_dq

    dfl = [node for node in graph.nodes if node.op == "Conv"][-1]  # DFL is the last Conv in the graph
    dfl.inputs[0] = dfl.i().i().inputs[0]  # bypass the input Q/DQ
    dfl.inputs[1] = dfl.i(1).i().inputs[0]  # bypass the weight Q/DQ
    graph.cleanup().toposort()
    onnx.save(gs.export_onnx(graph), save_file)
    print(f"Save onnx to {save_file}")

def run_export(weight, save, size, dynamic, noqadd, output, simplify, graphsurgeon, ort):
    if not noqadd and quantize is None:
        raise ImportError("--qadd needs pytorch_quantization installed")
    if quantize is not None:
        quantize.initialize()
    if save is None:
        name = os.path.basename(weight)
        name = name[:name.rfind('.')]
        save = os.path.join(os.path.dirname(weight), name + ".onnx")

    model = torch.load(weight, map_location="cpu")["model"]
    model.float()
    if not noqadd:
        quantize.replace_bottleneck_forward(model)
        quantize.apply_custom_rules_to_quantizer(model, export_onnx)

    export_onnx(model, save, size, dynamic, output, simplify, ort)

    if graphsurgeon:
        graphsurgeon_model(save)


