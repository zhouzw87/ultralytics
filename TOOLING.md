# Tooling

Quantization and deployment tooling layered on this checkout. Not part of upstream `ultralytics` — the package
itself is unmodified apart from `ultralytics/utils/export/rknn.py`, which gained an `rknn_wrapper` for
undecoded per-scale heads.

| Directory                         | What lives there                                                           |
| --------------------------------- | -------------------------------------------------------------------------- |
| [`trains/`](trains/README.md)     | `qat.py` — PTQ/QAT and ONNX export; the QAT core; training and tuning scripts |
| [`trts/`](trts/README.md)         | TensorRT-side evaluation and inference over a built `.engine`                |

Shortest path from a trained checkpoint to a TensorRT engine:

```bash
cd trains
python qat.py finetune --weight best.pt --data datasets/custom_cap.yaml   # PTQ + QAT, writes qat.pt
python qat.py export   --weight qat.pt --output raw                       # or end2end/rknn/split/transpose
trtexec --onnx=qat.onnx --saveEngine=qat.engine --fp16 --explicitBatch \
        --minShapes=images:1x3x640x640 --optShapes=images:16x3x640x640 --maxShapes=images:16x3x640x640
cd ../trts
python eval-trt.py --engine ../trains/qat.engine --data ../trains/datasets/custom_cap.yaml
```

Private dataset yamls live in `trains/datasets/`, deliberately outside `ultralytics/cfg/datasets/` so they do
not ship with the package or conflict with upstream.
