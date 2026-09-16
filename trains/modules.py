import os
import json
import torch
import torch.nn as nn
from copy import deepcopy

from ultralytics.cfg import get_cfg
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.downloads import attempt_download_asset as attempt_download
from ultralytics.nn.modules import Conv
from ultralytics.data.utils import check_det_dataset
from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.utils.torch_utils import init_seeds
from ultralytics.models import yolo

import quantize
from export import export_onnx

def setup_cfg(data, batch=16):
    """Seed and build the config plus the resolved dataset, per run rather than at import.

    Returns:
        (tuple): The config, and the dataset dict whose 'train'/'val' hold the resolved image directories.
    """
    init_seeds(2023)
    cfg = get_cfg()
    cfg.data, cfg.batch, cfg.mode = data, batch, "export"
    return cfg, check_det_dataset(data)


class SummaryTool:
    def __init__(self, file):
        self.file = file
        self.data = []

    def append(self, item):
        self.data.append(item)
        json.dump(self.data, open(self.file, "w"), indent=4)


# from ultralytics.engine.model import Model
def load_yolov8_model(weight, device, cfg) -> DetectionModel:
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


def evaluate_coco(model, val_dataloader, cfg):
    validator = yolo.detect.DetectionValidator(dataloader=val_dataloader, args=cfg)
    val_model = deepcopy(model)  # deepcopy
    mAP = validator(model=val_model)["metrics/mAP50-95(B)"]
    return mAP

def get_dataloader(cfg, data, img_path, mode='train', gs=32):
    """Build a dataloader over img_path; mode='val' disables augmentation and enables rectangular batching."""
    assert mode in ['train', 'val']
    dataset = build_yolo_dataset(cfg, img_path, cfg.batch, data, mode=mode, rect=mode == 'val', stride=gs)
    workers = cfg.workers if mode == 'train' else cfg.workers * 2
    return build_dataloader(dataset, cfg.batch, workers, shuffle=mode == 'train')



def run_sensitive_analysis(weight, data, device, summary_save):
    quantize.initialize()
    cfg, dataset = setup_cfg(data)
    device = torch.device(device)
    model = load_yolov8_model(weight, device, cfg)
    # Calibrate on train images, and in 'val' mode so augmentation does not skew the activation ranges
    calib_dataloader = get_dataloader(cfg, dataset, dataset['train'], mode='val')
    val_dataloader = get_dataloader(cfg, dataset, dataset['val'], mode='val')
    quantize.replace_to_quantization_module(model)
    quantize.calibrate_model(model, calib_dataloader, device)

    summary = SummaryTool(summary_save)
    print("Evaluate PTQ...")
    ap = evaluate_coco(model, val_dataloader, cfg)
    summary.append([ap, "PTQ"])

    print("Sensitive analysis by each layer...")
    for i in range(0, len(model.model)):
        layer = model.model[i]
        if quantize.have_quantizer(layer):
            print(f"Quantization disable model.{i}")
            quantize.disable_quantization(layer).apply()
            ap = evaluate_coco(model, val_dataloader, cfg)
            summary.append([ap, f"model.{i}"])
            quantize.enable_quantization(layer).apply()
        else:
            print(f"ignore model.{i} because it is {type(layer)}")

    summary = sorted(summary.data, key=lambda x: x[0], reverse=True)
    print("Sensitive summary:")
    for n, (ap, name) in enumerate(summary[:10]):
        print(f"Top{n}: Using fp16 {name}, ap = {ap:.5f}")

def run_qat(weight, data, device, ignore_policy, save_ptq, save_qat,
            supervision_stride, iters, eval_origin,eval_ptq):
    quantize.initialize()
    cfg, dataset = setup_cfg(data)

    if save_ptq and os.path.dirname(save_ptq) != "":
        os.makedirs(os.path.dirname(save_ptq), exist_ok=True)

    if save_qat and os.path.dirname(save_qat) != "":
        os.makedirs(os.path.dirname(save_qat), exist_ok=True)

    device = torch.device(device)
    print("Load model ....")
    model = load_yolov8_model(weight, device, cfg)
    print("Load dataset ....")
    train_dataloader = get_dataloader(cfg, dataset, dataset['train'], mode='train')
    val_dataloader = get_dataloader(cfg, dataset, dataset['val'], mode='val')
    # Calibrate on train images, and in 'val' mode so augmentation does not skew the activation ranges
    calib_dataloader = get_dataloader(cfg, dataset, dataset['train'], mode='val')
    print("Insert QDQ ....")
    quantize.replace_bottleneck_forward(model)
    if ignore_policy is None:  # Detect head index varies per model, so derive it instead of hardcoding
        ignore_policy = rf"model\.{len(model.model) - 1}\..*"
        print(f"Keep Detect head in high precision: {ignore_policy}")
    quantize.replace_to_quantization_module(model,ignore_policy)
    print("Apply custom_rules ....")
    quantize.apply_custom_rules_to_quantizer(model, export_onnx)
    print("Calibrate model ....")
    quantize.calibrate_model(model, calib_dataloader, device)

    json_save_dir = "." if os.path.dirname(save_ptq) == "" else os.path.dirname(save_ptq)
    summary_file = os.path.join(json_save_dir, "summary.json")
    summary = SummaryTool(summary_file)

    if eval_origin:
        print("Evaluate Origin...")
        with quantize.disable_quantization(model):
            ap = evaluate_coco(model, val_dataloader, cfg)
            summary.append(["Origin", ap])

    if eval_ptq:
        print("Evaluate PTQ...")
        ap = evaluate_coco(model, val_dataloader, cfg)
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
        ap = evaluate_coco(model, val_dataloader, cfg)
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


def run_test(weight, data, device):
    cfg, dataset = setup_cfg(data)
    device = torch.device(device)
    model = load_yolov8_model(weight, device, cfg)
    val_dataloader = get_dataloader(cfg, dataset, dataset['val'], mode='val')
    evaluate_coco(model, val_dataloader, cfg)