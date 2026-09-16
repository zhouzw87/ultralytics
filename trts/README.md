# trts

TensorRT-side tooling: evaluate and run inference on a built `.engine`. Building the ONNX that feeds these
lives in [`../trains`](../trains/README.md).

## Requirements

These scripts import `tensorrt` and `pycuda` at module level, so they only run on a machine with a working
TensorRT install. `eval-trt.py` also needs the repo's `ultralytics` importable:

```bash
cd trts
export PYTHONPATH=/path/to/repo
```

## The engine contract

Everything here decodes the **standard detection head**, `(1, 4+nc, 8400)`. Build the engine from an ONNX
exported with `--output raw`:

```bash
cd ../trains
python qat.py export --weight qat.pt --output raw
trtexec --onnx=qat.onnx --saveEngine=qat.engine --fp16 --explicitBatch \
        --minShapes=images:1x3x640x640 --optShapes=images:16x3x640x640 --maxShapes=images:16x3x640x640
```

An engine built from `end2end`, `split` or `rknn` will not match what these scripts decode. Engines exported
before the `Detect.export` fix carry three leaked feature maps alongside the real output; the scripts here read
the first binding so they tolerate it, but `ultralytics`' own `AutoBackend` sorts bindings by name and would
pick the wrong one. Re-export rather than reuse those.

## Files

| File               | What it does                                                                     |
| ------------------ | -------------------------------------------------------------------------------- |
| `eval-trt.py`      | mAP of an engine over a dataset's val split. Has a CLI: `--engine`, `--data`, … |
| `infer_trt.py`     | Run one engine over an image directory and write annotated images                 |
| `infer_trt_txt.py` | Ensemble three engines to write YOLO-format `.txt` labels, for pseudo-labelling   |
| `export_onnx.py`   | ONNX export through the stock `ultralytics` exporter, unrelated to the QAT path   |
| `metrics.py`       | `ap_per_class` / `ConfusionMatrix`, kept deliberately — see below                 |

```bash
python eval-trt.py --engine ../trains/qat.engine --data ../trains/datasets/custom_cap.yaml
```

`--data` drives the images, the class names and `nc`; the val split comes from its `val:` entry.

## Metric semantics

`metrics.py` is the YOLOv7-lineage implementation and is kept on purpose, so the numbers stay comparable with
this project's own history. It is **not** interchangeable with `ultralytics.utils.metrics`: the two append
different sentinels to the recall curve before the 101-point interpolation, and measure roughly 0.4–0.6 mAP
apart on the same inputs, with ultralytics reading lower. If you cross-check against
`YOLO(engine, task="detect").val(...)`, expect that offset and do not read it as a regression.

## Rough edges

- `infer_trt.py` and `infer_trt_txt.py` have no CLI. Engine paths, the image directory and the class list are
  constants at the top of each file; edit them before running.
- Both of those still carry the 80-class COCO name list, so annotated output is mislabelled for any other
  dataset. `eval-trt.py` no longer does — it reads names from `--data`.
