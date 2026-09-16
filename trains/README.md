# qat.py

PTQ/QAT and ONNX export for YOLO detection checkpoints, on top of this repo's `ultralytics` package.

```bash
python qat.py export    --weight best.pt --output rknn
python qat.py finetune  --weight best.pt --data datasets/custom_cap.yaml
python qat.py sensitive --weight best.pt --data datasets/custom_cap.yaml
python qat.py test      --weight best.pt --data datasets/custom_cap.yaml
```

Each mode lists only its own options: `python qat.py export --help`.

## Setup

Run everything from this `trains/` directory, and make sure `ultralytics` is importable:

```bash
cd trains
export PYTHONPATH=/path/to/repo      # or once: pip install -e /path/to/repo
```

`export` needs only `onnx`, `onnxsim` and `onnx_graphsurgeon`. The other three modes additionally need
`pytorch_quantization`, and `--graphsurgeon` needs it too since it only applies to a quantized graph.

## Modes

| Mode        | What it does                                                                            |
| ----------- | --------------------------------------------------------------------------------------- |
| `export`    | Trace the checkpoint to ONNX in one of the output layouts below                           |
| `finetune`  | Insert Q/DQ, calibrate, then QAT-finetune against the FP32 model, saving ptq.pt and qat.pt |
| `sensitive` | Disable quantization one layer at a time, ranking layers by the mAP recovered in FP16      |
| `test`      | Evaluate a checkpoint's mAP through the ultralytics validator                             |

`--data` takes a dataset yaml and is the single source for both the images and the class names; the val
split comes from its `val:` entry.

## Output layouts

`--output` picks the graph shape, one per deployment target. They are mutually exclusive.

| `--output`  | ONNX outputs                                         | For                                               |
| ----------- | ---------------------------------------------------- | ------------------------------------------------- |
| `end2end`   | `num_dets`, `det_boxes`, `det_scores`, `det_classes` | TensorRT with the EfficientNMS plugin (default)   |
| `raw`       | `output0` `(1, 4+nc, 8400)`                          | The standard decoded head; what `trts/` evaluates |
| `transpose` | `output0` `(1, 8400, 4+nc)`                          | Runtimes wanting the transposed layout            |
| `split`     | `bbox`, `conf`, `class_id`                           | Runtimes that run their own NMS                   |
| `rknn`      | `reg1`, `cls1`, … one pair per scale, undecoded      | RK-series NPUs, decode on the host CPU            |

`--ort` only applies to `end2end`, selecting ONNX Runtime NMS instead of the TensorRT plugin.

End-to-end checkpoints (YOLO26, YOLOv10) already return `(batch, max_det, 6)`, so they only accept
`raw` and `rknn`; the other three raise rather than silently emitting a wrong graph.

## Typical flows

Plain export, no quantization:

```bash
python qat.py export --weight runs/detect/cap1.2/train/weights/best.pt --output raw
```

Quantize and export for TensorRT:

```bash
python qat.py sensitive --weight best.pt --data datasets/custom_cap.yaml   # optional, find fragile layers
python qat.py finetune  --weight best.pt --data datasets/custom_cap.yaml
python qat.py export    --weight qat.pt --output end2end --graphsurgeon
trtexec --onnx=qat.onnx --saveEngine=qat.engine --fp16 --explicitBatch \
        --minShapes=images:1x3x640x640 --optShapes=images:16x3x640x640 --maxShapes=images:16x3x640x640
```

By default the Detect head is kept in high precision, since quantizing it costs accuracy. Override with
`--ignore-policy` once `sensitive` tells you which other layers to spare, e.g. `--ignore-policy 'model\.(9|22)\..*'`.

## Notes

- `--graphsurgeon` rewrites the exported file in place, sharing Q/DQ across an activation's branches so
  TensorRT can fuse it. It only applies to a quantized graph and errors out on any other.
- Calibration uses train images with augmentation off. Calibrating on the val split would leak into the
  reported mAP.
- `trts/eval-trt.py` evaluates a built `.engine` and expects `--output raw`. An engine built from any
  other layout will not match what it decodes.
