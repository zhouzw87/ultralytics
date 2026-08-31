import os
import torch.nn
import torch.optim as optim
from torch.cuda import amp

from copy import deepcopy
from tqdm import tqdm
from typing import Dict
from absl import logging as quant_logging

from pytorch_quantization import quant_modules
from pytorch_quantization import tensor_quant

from rule_units import find_quantizer_pairs
from quantize_units import *


class disable_quantization:
    def __init__(self, model):
        self.model = model

    def apply(self, disabled=True):
        for name, module in self.model.named_modules():
            if isinstance(module, quant_nn.TensorQuantizer):
                module._disabled = disabled

    def __enter__(self):
        self.apply(True)

    def __exit__(self, *args, **kwargs):
        self.apply(False)


class enable_quantization:
    def __init__(self, model):
        self.model = model

    def apply(self, enabled=True):
        for name, module in self.model.named_modules():
            if isinstance(module, quant_nn.TensorQuantizer):
                module._disabled = not enabled

    def __enter__(self):
        self.apply(True)
        return self

    def __exit__(self, *args, **kwargs):
        self.apply(False)


def have_quantizer(module):
    for name, module in module.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            return True


# Initialize PyTorch Quantization
def initialize():
    quant_desc_input = QuantDescriptor(calib_method="histogram")
    quant_nn.QuantConv2d.set_default_quant_desc_input(quant_desc_input)
    quant_nn.QuantMaxPool2d.set_default_quant_desc_input(quant_desc_input)
    quant_nn.QuantLinear.set_default_quant_desc_input(quant_desc_input)
    quant_logging.set_verbosity(quant_logging.ERROR)

# For example: YoloV5 Bottleneck
def replace_bottleneck_forward(model):
    # for name, bottleneck in model.named_modules():
    #     if bottleneck.__class__.__name__ == "Bottleneck":
    #         if bottleneck.add:
    #             if not hasattr(bottleneck, "addop"):
    #                 print(f"Add QuantAdd to {name}")
    #                 bottleneck.addop = QuantAdd(bottleneck.add)
    #             bottleneck.__class__.forward = bottleneck_quant_forward
    for name, module in model.named_modules():
        if module.__class__.__name__ == "C2f":
            if not hasattr(module, "c2fchunkop"):
                print(f"Add C2fQuantChunk to {name}")
                module.c2fchunkop = QuantC2fChunk(module.c)
            module.__class__.forward = c2f_qaunt_forward

        if module.__class__.__name__ == "Bottleneck":
            if module.add:
                if not hasattr(module, "addop"):
                    print(f"Add QuantAdd to {name}")
                    module.addop = QuantAdd(module.add)
                module.__class__.forward = bottleneck_quant_forward

        if module.__class__.__name__ == "Concat":
            if not hasattr(module, "concatop"):
                print(f"Add QuantConcat to {name}")
                module.concatop = QuantConcat(module.d)
            module.__class__.forward = concat_quant_forward

        if module.__class__.__name__ == "Upsample":
            if not hasattr(module, "upsampleop"):
                print(f"Add QuantUpsample to {name}")
                module.upsampleop = QuantUpsample(module.size, module.scale_factor, module.mode)
            module.__class__.forward = upsample_quant_forward


def replace_to_quantization_module(model: torch.nn.Module, ignore_policy: Union[str, List[str], Callable] = None):
    module_dict = {}
    # print("_DEFAULT_QUANT_MAP:",quant_modules._DEFAULT_QUANT_MAP)
    for entry in quant_modules._DEFAULT_QUANT_MAP:
        module = getattr(entry.orig_mod, entry.mod_name)
        module_dict[id(module)] = entry.replace_mod
    print("module_dict:",module_dict)
    def recursive_and_replace_module(module, prefix=""):
        for name in module._modules:
            submodule = module._modules[name]
            # print("module._modules:", name,submodule)
            path = name if prefix == "" else prefix + "." + name
            recursive_and_replace_module(submodule, path)

            submodule_id = id(type(submodule))
            if submodule_id in module_dict:
                ignored = quantization_ignore_match(ignore_policy, path)
                if ignored:
                    print(f"Quantization: {path} has ignored.")
                    continue
                # print("module._modules[name]:", module._modules[name])
                module._modules[name] = transfer_torch_to_quantization(submodule, module_dict[submodule_id])
            # else:
            #     print("submodule_id:", module._modules[name])
    recursive_and_replace_module(model)

def apply_custom_rules_to_quantizer(model: torch.nn.Module, export_onnx: Callable):
    # apply rules to graph
    export_onnx(model, "quantization-custom-rules-temp.onnx")
    pairs = find_quantizer_pairs("quantization-custom-rules-temp.onnx")
    for major, sub in pairs:
        print(f"Rules: {sub} match to {major}")
        get_attr_with_path(model, sub)._input_quantizer = get_attr_with_path(model, major)._input_quantizer
    os.remove("quantization-custom-rules-temp.onnx")

    for name, module in model.named_modules():
        if module.__class__.__name__ == "Bottleneck":
            if module.add:
                print(f"Rules: {name}.add match to {name}.cv1")
                major = module.cv1.conv._input_quantizer
                module.addop._input0_quantizer = major
                module.addop._input1_quantizer = major
        # if module.__class__.__name__ == "Bottleneck":
        #     if module.add:
        #         if not hasattr(module, "addop"):
        #             print(f"Add QuantAdd to {name}")
        #             module.addop = QuantAdd(module.add)
        #         module.__class__.forward = bottleneck_quant_forward

def calibrate_model(model: torch.nn.Module, dataloader, device, num_batch=1024):
    def compute_amax(model, **kwargs):
        for name, module in model.named_modules():
            if isinstance(module, quant_nn.TensorQuantizer):
                if module._calibrator is not None:
                    if isinstance(module._calibrator, calib.MaxCalibrator):
                        module.load_calib_amax()
                    else:
                        module.load_calib_amax(**kwargs)

                    module._amax = module._amax.to(device)

    def collect_stats(model, data_loader, device, num_batch=200):
        """Feed data to the network and collect statistics"""
        # Enable calibrators
        model.eval()
        for name, module in model.named_modules():
            if isinstance(module, quant_nn.TensorQuantizer):
                if module._calibrator is not None:
                    module.disable_quant()
                    module.enable_calib()
                else:
                    module.disable()

        # Feed data to the network for collecting stats
        with torch.no_grad():
            for i, datas in tqdm(enumerate(data_loader), total=num_batch, desc="Collect stats for calibrating"):
                imgs = datas['img'].to(device, non_blocking=True).float() / 255
                model(imgs)

                if i >= num_batch:
                    break

        # Disable calibrators
        for name, module in model.named_modules():
            if isinstance(module, quant_nn.TensorQuantizer):
                if module._calibrator is not None:
                    module.enable_quant()
                    module.disable_calib()
                else:
                    module.enable()

    collect_stats(model, dataloader, device, num_batch=num_batch)
    compute_amax(model, method="mse")

def export_onnx(model, input, file, *args, **kwargs):
    quant_nn.TensorQuantizer.use_fb_fake_quant = True

    model.eval()
    with torch.no_grad():
        torch.onnx.export(model, input, file, *args, **kwargs)

    quant_nn.TensorQuantizer.use_fb_fake_quant = False
    
def finetune(model: torch.nn.Module, train_dataloader, per_epoch_callback: Callable = None, preprocess: Callable = None,
            nepochs=10, early_exit_batchs_per_epoch=1024, lrschedule: Dict = None, fp16=True, learningrate=1e-4,
            supervision_policy: Callable = None):
    origin_model = deepcopy(model).eval()
    disable_quantization(origin_model).apply()

    model.train()
    model.requires_grad_(True)

    scaler = amp.GradScaler(enabled=fp16)
    optimizer = optim.Adam(model.parameters(), learningrate)
    quant_lossfn = torch.nn.MSELoss()
    device = next(model.parameters()).device

    if lrschedule is None:
        lrschedule = {
            0: 1e-4,
            2: 1e-5,
            4: 1e-6,
            8: 1e-7,
        }

    def make_layer_forward_hook(l):
        def forward_hook(m, input, output):
            l.append(output)

        return forward_hook

    supervision_module_pairs = []
    for ((mname, ml), (oriname, ori)) in zip(model.named_modules(), origin_model.named_modules()):
        if isinstance(ml, quant_nn.TensorQuantizer): continue

        if supervision_policy:
            if not supervision_policy(mname, ml):
                continue

        supervision_module_pairs.append([ml, ori])

    for iepoch in range(nepochs):

        if iepoch in lrschedule:
            learningrate = lrschedule[iepoch]
            for g in optimizer.param_groups:
                g["lr"] = learningrate

        model_outputs = []
        origin_outputs = []
        remove_handle = []

        for ml, ori in supervision_module_pairs:
            remove_handle.append(ml.register_forward_hook(make_layer_forward_hook(model_outputs)))
            remove_handle.append(ori.register_forward_hook(make_layer_forward_hook(origin_outputs)))

        model.train()
        pbar = tqdm(train_dataloader, desc="QAT", total=early_exit_batchs_per_epoch)
        for ibatch, imgs in enumerate(pbar):

            if ibatch >= early_exit_batchs_per_epoch:
                break

            if preprocess:
                imgs = preprocess(imgs, device)

            # imgs = imgs.to(device)
            with amp.autocast(enabled=fp16):
                model(imgs)

                with torch.no_grad():
                    origin_model(imgs)

                quant_loss = 0
                for index, (mo, fo) in enumerate(zip(model_outputs, origin_outputs)):
                    quant_loss += quant_lossfn(mo, fo)

                model_outputs.clear()
                origin_outputs.clear()

            if fp16:
                scaler.scale(quant_loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                quant_loss.backward()
                optimizer.step()
            optimizer.zero_grad()
            pbar.set_description(
                f"QAT Finetuning {iepoch + 1} / {nepochs}, Loss: {quant_loss.detach().item():.5f}, LR: {learningrate:g}")

        # You must remove hooks during onnx export or torch.save
        for rm in remove_handle:
            rm.remove()

        if per_epoch_callback:
            if per_epoch_callback(model, iepoch, learningrate):
                break
