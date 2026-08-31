import os
import json
import torch
import torch.nn as nn
import onnx
import onnxsim
import onnx_graphsurgeon as gs
from copy import deepcopy

from ultralytics.cfg import get_cfg
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.downloads import attempt_download_asset as attempt_download
from ultralytics.nn.modules import Conv
from ultralytics.data.utils import check_cls_dataset, check_det_dataset
from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.utils import clean_url, emojis
from ultralytics.utils import LOGGER
from ultralytics.utils.torch_utils import init_seeds
from ultralytics.models import yolo
from ultralytics.utils.checks import check_imgsz, check_requirements
from ultralytics.utils import colorstr

import quantize

init_seeds(2023)
cfg = get_cfg("../ultralytics/cfg/default.yaml")
cfg.data = "../ultralytics/cfg/datasets/custom_peoplecar.yaml"
cfg.batch = 16
cfg.mode = "export"
print("torch.__version__:  ", torch.__version__)

class SummaryTool:
    def __init__(self, file):
        self.file = file
        self.data = []

    def append(self, item):
        self.data.append(item)
        json.dump(self.data, open(self.file, "w"), indent=4)


# from ultralytics.engine.model import Model
def load_yolov8_model(weight, device) -> DetectionModel:
    attempt_download(weight)
    model = torch.load(weight, map_location=device)["model"]
    for m in model.modules():
        if type(m) is nn.Upsample:
            m.recompute_scale_factor = None  # torch 1.11.0 compatibility
        elif type(m) is Conv:
            m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility
    model.args = cfg  # model.args : 字典类型  改为  cfg: IterableSimpleNamespace类型
    model.float()
    model.eval()

    with torch.no_grad():
        model.fuse()
    return model


def build_dataset(cfg, img_path, mode='train', batch=None, gs=32):
    """Build YOLO Dataset."""
    try:
        if cfg.task == 'classify':
            data = check_cls_dataset(cfg.data)
        elif cfg.data.split('.')[-1] in ('yaml', 'yml') or cfg.task in ('detect', 'segment', 'pose'):
            data = check_det_dataset(cfg.data)
            if 'yaml_file' in data:
                cfg.data = data['yaml_file']  # for validating 'yolo train data=url.zip' usage
    except Exception as e:
        raise RuntimeError(emojis(f"Dataset '{clean_url(cfg.data)}' error ❌ {e}")) from e
    return build_yolo_dataset(cfg, img_path, batch, data, mode=mode, rect=mode == 'val', stride=gs)

def evaluate_coco(model, val_dataloader):
    validator = yolo.detect.DetectionValidator(dataloader=val_dataloader, args=cfg)
    val_model = deepcopy(model)  # deepcopy
    mAP = validator(model=val_model)["metrics/mAP50-95(B)"]
    return mAP

def get_dataloader(cfg, dataset_path, batch_size=16, mode='train', gs=32):
    assert mode in ['train', 'val']
    # with torch_distributed_zero_first(rank):  # init dataset *.cache only once if DDP
    dataset = build_dataset(cfg, dataset_path, mode, batch=batch_size, gs=gs)
    shuffle = mode == 'train'
    if getattr(dataset, 'rect', False) and shuffle:
        LOGGER.warning("WARNING ⚠️ 'rect=True' is incompatible with DataLoader shuffle, setting shuffle=False")
        shuffle = False
    workers = cfg.workers if mode == 'train' else cfg.workers * 2
    return build_dataloader(dataset, batch_size, workers, shuffle)  # return dataloader



def export_onnx(model: DetectionModel, save_file, size=640, dynamic_batch=False,
                end2end=False,simplify=False, ort=False,rknn=False, transoutput=False,prefix=colorstr('ONNX:')):
    requirements = ['onnx>=1.12.0']
    check_requirements(requirements)

    device = next(model.parameters()).device

    batch_size = 1
    imgsz = check_imgsz(cfg.imgsz, stride=model.stride, min_dim=2)
    input = torch.zeros(batch_size, 3, size, size).to(device)

    dynamic_axes = None
    if dynamic_batch:
        batch_size = 'batch'
        dynamic_axes = { 'images' :{0:'batch',2:"height",3:"width",}}
        if end2end:
            output_axes = { 'num_dets': {0: 'batch'},'det_boxes': {0: 'batch'},'det_scores': {0: 'batch'},'det_classes': {0: 'batch'}}
        elif rknn:
            output_axes = { 'reg1': {0: 'batch'},'cls1': {0: 'batch'},'reg2': {0: 'batch'},'cls2': {0: 'batch'}, 'reg3': {0: 'batch'},'cls3': {0: 'batch'}}
        else:
            output_axes = {'outputs': {0: 'batch'}, }
        dynamic_axes.update(output_axes)
    if end2end:
        output_names = ['num_dets', 'det_boxes', 'det_scores', 'det_classes']
    elif rknn:
        output_names = ['reg1', 'cls1', 'reg2', 'cls2','reg3', 'cls3']
    else:
        output_names =  ['output0']

    if end2end:
        from end2end import End2End
        model = End2End(model, max_obj=100, iou_thres=0.45,score_thres=0.5,
                        device=device, ort=ort, trt_version=8, with_preprocess=False)
    if end2end==False and transoutput:
        from end2end import TransOut
        model = TransOut(model, device=device)
    quantize.export_onnx(model.cpu() if dynamic_batch else model,  # dynamic=True only compatible with cpu
                         input.cpu() if dynamic_batch else input,
                         save_file,verbose=False,opset_version=13,do_constant_folding=True,
                         input_names=['images'],
                         output_names=output_names,
                         dynamic_axes=dynamic_axes
                         )
    # Simplify
    onnx_model = onnx.load(save_file)  # load onnx model
    onnx.checker.check_model(onnx_model)  # check onnx model
    # Fix output shape
    if end2end and not ort:
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


def graphsurgeon_model(onnx_model):
    graph = gs.import_onnx(onnx.load(onnx_model))
    nodes = graph.nodes
    mul_nodes = [node for node in graph.nodes if node.op == "Mul"]
    # mul_nodes = [node for node in graph.nodes if
    #              node.op == "Mul" and node.i(0).op == "BatchNormalization" and node.i(1).op == "Sigmoid"]

    many_outputs_mul_nodes = []
    for node in mul_nodes:  # convolution mul node for silu activation.
        try:
            for i in range(99):
                node.o(i)
        except:
            if i > 1:
                mul_nodename_outnum = {"node": node, "out_num": i}
                many_outputs_mul_nodes.append(mul_nodename_outnum)

    for node_dict in many_outputs_mul_nodes:
        if node_dict["out_num"] == 2:
            if node_dict["node"].o(0).op == "QuantizeLinear" and node_dict["node"].o(1).op == "QuantizeLinear":
                if node_dict["node"].o(1).o(0).o(0).op == "Concat":
                    concat_dq_out_name = node_dict["node"].o(1).o(0).outputs[0].name
                    for i, concat_input in enumerate(node_dict["node"].o(1).o(0).o(0).inputs):
                        if concat_input.name == concat_dq_out_name:
                            node_dict["node"].o(1).o(0).o(0).inputs[i] = node_dict["node"].o(0).o(0).outputs[0]  # concat 4개
                else:
                    node_dict["node"].o(1).o(0).o(0).inputs[0] = node_dict["node"].o(0).o(0).outputs[0]  # 그 외

            elif node_dict["node"].o(0).op == "QuantizeLinear" and node_dict["node"].o(1).op == "Concat":
                concat_dq_out_name = node_dict["node"].outputs[0].outputs[0].inputs[0].name
                for i, concat_input in enumerate(node_dict["node"].outputs[0].outputs[1].inputs):
                    if concat_input.name == concat_dq_out_name:
                        node_dict["node"].outputs[0].outputs[1].inputs[i] = \
                        node_dict["node"].outputs[0].outputs[0].o().outputs[0]  # concat 4개

        elif node_dict["out_num"] == 3:
            node_dict["node"].o(2).o(0).o(0).inputs[0] = node_dict["node"].o(0).o(0).outputs[0]
            node_dict["node"].o(1).o(0).o(0).inputs[0] = node_dict["node"].o(0).o(0).outputs[0]

        elif node_dict["out_num"] == 4:  # shape node not merged
            node_dict["node"].o(3).o(0).o(0).inputs[0] = node_dict["node"].o(0).o(0).outputs[0]
            node_dict["node"].o(2).o(0).o(0).inputs[0] = node_dict["node"].o(0).o(0).outputs[0]

    # add_nodes = [node for node in graph.nodes if node.op == "Add"]
    # many_outputs_add_nodes = []
    # for node in add_nodes:  # convolution mul node for silu activation.
    #     try:
    #         for i in range(99):
    #             node.o(i)
    #     except:
    #         if i > 1 and node.o().op == "QuantizeLinear":
    #             add_nodename_outnum = {"node": node, "out_num": i}
    #             many_outputs_add_nodes.append(add_nodename_outnum)
    #
    # for node_dict in many_outputs_add_nodes:
    #     if node_dict["node"].outputs[0].outputs[0].op == "QuantizeLinear" and node_dict["node"].outputs[0].outputs[2].op == "Concat":
    #         concat_dq_out_name = node_dict["node"].outputs[0].outputs[0].inputs[0].name
    #         for i, concat_input in enumerate(node_dict["node"].outputs[0].outputs[1].inputs):
    #             if concat_input.name == concat_dq_out_name:
    #                 node_dict["node"].outputs[0].outputs[1].inputs[i] = \
    #                 node_dict["node"].outputs[0].outputs[0].o().outputs[0]  # concat 4개

    conv_nodes = [node for node in graph.nodes if node.op == "Conv"]
    conv_nodes[-1].inputs[0] = conv_nodes[-1].i().i().inputs[0]  # dfl block input
    conv_nodes[-1].inputs[1] = conv_nodes[-1].i(1).i().inputs[0]  # dfl block weight
    graph.cleanup().toposort()
    onnx.save(gs.export_onnx(graph), "modified.onnx")
    print(f"Save onnx to modified.onnx")

def run_export(weight, save, size, dynamic, noqadd,end2end,simplify,graphsurgeon,ort,rknn,transoutput):
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

    export_onnx(model, save, size,dynamic,end2end,simplify,ort,rknn,transoutput)

    if graphsurgeon:
        graphsurgeon_model(save)


def run_sensitive_analysis(weight, device, cocodir, summary_save):
    quantize.initialize()
    device = torch.device(device)
    model = load_yolov8_model(weight, device)
    train_dataloader = get_dataloader(cfg, cocodir + "images/train", batch_size=cfg.batch, mode='train')
    val_dataloader = get_dataloader(cfg, cocodir + "images/val", batch_size=cfg.batch, mode='val')
    quantize.replace_to_quantization_module(model)
    quantize.calibrate_model(model, train_dataloader, device)

    summary = SummaryTool(summary_save)
    print("Evaluate PTQ...")
    ap = evaluate_coco(model, val_dataloader)
    summary.append([ap, "PTQ"])

    print("Sensitive analysis by each layer...")
    for i in range(0, len(model.model)):
        layer = model.model[i]
        if quantize.have_quantizer(layer):
            print(f"Quantization disable model.{i}")
            quantize.disable_quantization(layer).apply()
            ap = evaluate_coco(model, val_dataloader)
            summary.append([ap, f"model.{i}"])
            quantize.enable_quantization(layer).apply()
        else:
            print(f"ignore model.{i} because it is {type(layer)}")

    summary = sorted(summary.data, key=lambda x: x[0], reverse=True)
    print("Sensitive summary:")
    for n, (ap, name) in enumerate(summary[:10]):
        print(f"Top{n}: Using fp16 {name}, ap = {ap:.5f}")

def run_qat(weight, cocodir, device, ignore_policy, save_ptq, save_qat,
            supervision_stride, iters, eval_origin,eval_ptq):
    quantize.initialize()

    if save_ptq and os.path.dirname(save_ptq) != "":
        os.makedirs(os.path.dirname(save_ptq), exist_ok=True)

    if save_qat and os.path.dirname(save_qat) != "":
        os.makedirs(os.path.dirname(save_qat), exist_ok=True)

    device = torch.device(device)
    print("Load model ....")
    model = load_yolov8_model(weight, device)
    print("Load dataset ....")
    train_dataloader = get_dataloader(cfg, cocodir + "images/train", batch_size=cfg.batch, mode='train')
    val_dataloader = get_dataloader(cfg, cocodir + "images/val", batch_size=cfg.batch, mode='val')
    print("Insert QDQ ....")
    quantize.replace_bottleneck_forward(model)
    quantize.replace_to_quantization_module(model,ignore_policy)
    print("Apply custom_rules ....")
    quantize.apply_custom_rules_to_quantizer(model, export_onnx)
    print("Calibrate model ....")
    quantize.calibrate_model(model, val_dataloader, device)

    json_save_dir = "." if os.path.dirname(save_ptq) == "" else os.path.dirname(save_ptq)
    summary_file = os.path.join(json_save_dir, "summary.json")
    summary = SummaryTool(summary_file)

    if eval_origin:
        print("Evaluate Origin...")
        with quantize.disable_quantization(model):
            ap = evaluate_coco(model, val_dataloader)
            summary.append(["Origin", ap])

    if eval_ptq:
        print("Evaluate PTQ...")
        ap = evaluate_coco(model, val_dataloader)
        summary.append(["PTQ", ap])

    if save_ptq:
        print(f"Save ptq model to {save_ptq}")
        torch.save({"model": model}, save_ptq)

    if save_qat is None:
        print("Done as save_qat is None.")
        return

    best_ap = 0
    def per_epoch(model, epoch, lr):
        nonlocal best_ap
        ap = evaluate_coco(model, val_dataloader)
        summary.append([f"QAT{epoch}", ap])

        if ap > best_ap:
            print(f"Save qat model to {save_qat} @ {ap:.5f}")
            best_ap = ap
            torch.save({"model": model}, save_qat)

    torch.save({"model": model}, save_qat)
    # def preprocess(datas):
    #     return datas[0].to(device).float() / 255.0

    def preprocess_batch(batch, device):
        batch['img'] = batch['img'].to(device, non_blocking=True).float() / 255
        return batch

    def supervision_policy():
        supervision_list = []
        for item in model.model:
            supervision_list.append(id(item))
        keep_idx = list(range(0, len(model.model) - 1, supervision_stride))
        keep_idx.append(len(model.model) - 2)
        def impl(name, module):
            if id(module) not in supervision_list: return False
            idx = supervision_list.index(id(module))
            if idx in keep_idx:
                print(f"Supervision: {name} will compute loss with origin model during QAT training")
            else:
                print(
                    f"Supervision: {name} no compute loss during QAT training, that is unsupervised only and doesn't mean don't learn")
            return idx in keep_idx
        return impl

    quantize.finetune(model, train_dataloader, per_epoch, early_exit_batchs_per_epoch=iters,
        preprocess=preprocess_batch, supervision_policy=supervision_policy())


def run_test(weight, device, cocodir):
    device = torch.device(device)
    model = load_yolov8_model(weight, device)
    val_dataloader = get_dataloader(cfg, cocodir + "images/val", batch_size=cfg.batch, mode='val')
    evaluate_coco(model, val_dataloader)