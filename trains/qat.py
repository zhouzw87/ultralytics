"""PTQ/QAT and ONNX export for YOLO detection checkpoints.

    python qat.py export    --weight best.pt --output rknn
    python qat.py finetune  --weight best.pt --data datasets/custom_cap.yaml
    python qat.py sensitive --weight best.pt --data datasets/custom_cap.yaml
    python qat.py test      --weight best.pt --data datasets/custom_cap.yaml

Run `python qat.py <mode> --help` for the options a mode accepts.
"""

import argparse

from export import OUTPUT_NAMES, run_export
from modules import run_qat, run_sensitive_analysis, run_test

TRTEXEC_HINT = """build an engine from the exported onnx with:
  trtexec --onnx=qat.onnx --saveEngine=qat.engine --fp16 --explicitBatch \\
          --minShapes=images:1x3x640x640 --optShapes=images:16x3x640x640 --maxShapes=images:16x3x640x640"""


def build_parser():
    """Build the CLI, one subcommand per mode so each --help lists only what that mode uses."""
    weight = argparse.ArgumentParser(add_help=False)  # every mode loads a checkpoint
    weight.add_argument("--weight", type=str, required=True, help="checkpoint to load (yolov8/yolo11/yolo26)")

    dataset = argparse.ArgumentParser(add_help=False)  # every mode but export reads a dataset
    dataset.add_argument("--data", type=str, required=True,
                         help="dataset yaml supplying the images and class names, e.g. datasets/custom_cap.yaml")
    dataset.add_argument("--device", type=str, default="cuda:0", help="device")

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    modes = parser.add_subparsers(dest="mode", required=True, metavar="{export,finetune,sensitive,test}")

    export = modes.add_parser("export", parents=[weight], help="export the checkpoint to ONNX",
                              epilog=TRTEXEC_HINT, formatter_class=argparse.RawDescriptionHelpFormatter)
    export.add_argument("--save", type=str, default=None, help="output .onnx path, defaults to beside the weight")
    export.add_argument("--size", type=int, default=640, help="input size for export onnx")
    export.add_argument("--output", choices=sorted(OUTPUT_NAMES), default="end2end",
                        help="output layout, one per deployment target (default: %(default)s)")
    export.add_argument("--no-dynamic", dest="dynamic", action="store_false", help="export a fixed batch size")
    export.add_argument("--no-simplify", dest="simplify", action="store_false", help="skip onnxsim")
    export.add_argument("--qadd", action="store_true", help="add QuantAdd to residual connections")
    export.add_argument("--graphsurgeon", action="store_true", help="run graphsurgeon on the exported onnx")
    export.add_argument("--ort", action="store_true", help="output=end2end only: ONNX Runtime NMS instead of TensorRT")

    finetune = modes.add_parser("finetune", parents=[weight, dataset], help="calibrate, then QAT finetune")
    finetune.add_argument("--ptq", type=str, default="ptq.pt", help="PTQ checkpoint to save")
    finetune.add_argument("--qat", type=str, default="qat.pt", help="QAT checkpoint to save")
    finetune.add_argument("--no-eval-origin", dest="eval_origin", action="store_false",
                          help="skip the FP32 baseline eval")
    finetune.add_argument("--no-eval-ptq", dest="eval_ptq", action="store_false", help="skip the PTQ eval")
    finetune.add_argument("--ignore-policy", type=str, default=None,
                          help="regx of modules to keep in high precision, defaults to the Detect head")
    finetune.add_argument("--supervision-stride", type=int, default=1, help="supervision stride")
    finetune.add_argument("--iters", type=int, default=3700, help="iters per epoch")

    sensitive = modes.add_parser("sensitive", parents=[weight, dataset],
                                 help="per-layer quantization sensitivity analysis")
    sensitive.add_argument("--summary", type=str, default="sensitive-summary.json", help="summary save file")

    modes.add_parser("test", parents=[weight, dataset], help="evaluate the checkpoint's mAP")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()

    if args.mode == "export":
        run_export(args.weight, args.save, args.size, args.dynamic, not args.qadd,
                   args.output, args.simplify, args.graphsurgeon, args.ort)
    elif args.mode == "finetune":
        run_qat(args.weight, args.data, args.device, args.ignore_policy,
                args.ptq, args.qat, args.supervision_stride, args.iters,
                args.eval_origin, args.eval_ptq)
    elif args.mode == "sensitive":
        run_sensitive_analysis(args.weight, args.data, args.device, args.summary)
    else:
        run_test(args.weight, args.data, args.device)
