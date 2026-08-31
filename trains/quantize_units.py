import re
import torch
from torch.nn import functional as F
from pytorch_quantization import nn as quant_nn
from pytorch_quantization.nn.modules import _utils as quant_nn_utils
from pytorch_quantization import calib
from pytorch_quantization.tensor_quant import QuantDescriptor
from typing import List, Callable, Union


# class QuantAdd(torch.nn.Module):
#     def __init__(self, quantization):
#         super().__init__()
#
#         if quantization:
#             self._input0_quantizer = quant_nn.TensorQuantizer(QuantDescriptor(num_bits=8, calib_method="histogram"))
#             self._input1_quantizer = quant_nn.TensorQuantizer(QuantDescriptor(num_bits=8, calib_method="histogram"))
#             self._input0_quantizer._calibrator._torch_hist = True
#             self._input1_quantizer._calibrator._torch_hist = True
#             self._fake_quant = True
#         self.quantization = quantization
#
#     def forward(self, x, y):
#         if self.quantization:
#             # print(f"QAdd {self._input0_quantizer}  {self._input1_quantizer}")
#             return self._input0_quantizer(x) + self._input1_quantizer(y)
#         return x + y


class QuantAdd(torch.nn.Module):
    def __init__(self, quantization):
        super().__init__()
        if quantization:
            self._input0_quantizer = quant_nn.TensorQuantizer(QuantDescriptor())
            self._input1_quantizer = quant_nn.TensorQuantizer(QuantDescriptor())
        self.quantization = quantization
        self._fake_quant = True

    def forward(self, x, y):
        if self.quantization:
            # print(f"QAdd {self._input0_quantizer}  {self._input1_quantizer}")
            return self._input0_quantizer(x) + self._input1_quantizer(y)
        return x + y


class QuantC2fChunk(torch.nn.Module):
    def __init__(self, c):
        super().__init__()
        self._input0_quantizer = quant_nn.TensorQuantizer(QuantDescriptor())
        self.c = c

    def forward(self, x, chunks, dims):
        return torch.split(self._input0_quantizer(x), (self.c, self.c), dims)

class QuantConcat(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self._input0_quantizer = quant_nn.TensorQuantizer(QuantDescriptor())
        self._input1_quantizer = quant_nn.TensorQuantizer(QuantDescriptor())
        self.dim = dim

    def forward(self, x, dim):
        x_0 = self._input0_quantizer(x[0])
        x_1 = self._input1_quantizer(x[1])
        return torch.cat((x_0, x_1), self.dim)


class QuantUpsample(torch.nn.Module):
    def __init__(self, size, scale_factor, mode):
        super().__init__()
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        self._input_quantizer = quant_nn.TensorQuantizer(QuantDescriptor())

    def forward(self, x):
        return F.interpolate(self._input_quantizer(x), self.size, self.scale_factor, self.mode)


def bottleneck_quant_forward(self, x):
    if hasattr(self, "addop"):
        return self.addop(x, self.cv2(self.cv1(x))) if self.add else self.cv2(self.cv1(x))
    return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

def concat_quant_forward(self, x):
    if hasattr(self, "concatop"):
        return self.concatop(x, self.d)
    return torch.cat(x, self.d)

def upsample_quant_forward(self, x):
    if hasattr(self, "upsampleop"):
        return self.upsampleop(x)
    return F.interpolate(x)

def c2f_qaunt_forward(self, x):
    if hasattr(self, "c2fchunkop"):
        y = list(self.c2fchunkop(self.cv1(x), 2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    else:
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

def transfer_torch_to_quantization(nninstance: torch.nn.Module, quantmodule):
    quant_instance = quantmodule.__new__(quantmodule)
    for k, val in vars(nninstance).items():
        setattr(quant_instance, k, val)

    def __init__(self):
        quant_desc_input, quant_desc_weight = quant_nn_utils.pop_quant_desc_in_kwargs(self.__class__)
        if isinstance(self, quant_nn_utils.QuantInputMixin):
            self.init_quantizer(quant_desc_input)

            # Turn on torch_hist to enable higher calibration speeds
            if isinstance(self._input_quantizer._calibrator, calib.HistogramCalibrator):
                self._input_quantizer._calibrator._torch_hist = True
        else:
            self.init_quantizer(quant_desc_input, quant_desc_weight)

            # Turn on torch_hist to enable higher calibration speeds
            if isinstance(self._input_quantizer._calibrator, calib.HistogramCalibrator):
                self._input_quantizer._calibrator._torch_hist = True
                self._weight_quantizer._calibrator._torch_hist = True

    __init__(quant_instance)
    return quant_instance

def quantization_ignore_match(ignore_policy: Union[str, List[str], Callable], path: str) -> bool:
    if ignore_policy is None: return False
    if isinstance(ignore_policy, Callable):
        return ignore_policy(path)

    if isinstance(ignore_policy, str) or isinstance(ignore_policy, List):

        if isinstance(ignore_policy, str):
            ignore_policy = [ignore_policy]

        if path in ignore_policy: return True
        for item in ignore_policy:
            if re.match(item, path):
                return True
    return False

def get_attr_with_path(m, path):
    def sub_attr(m, names):
        name = names[0]
        # print(name)
        value = getattr(m, name)

        if len(names) == 1:
            return value

        return sub_attr(value, names[1:])

    return sub_attr(m, path.split("."))
