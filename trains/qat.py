import argparse

from modules import OUTPUT_NAMES, run_export, run_qat, run_sensitive_analysis, run_test

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PTQ/QAT and ONNX export for YOLO detection checkpoints")
    parser.add_argument("mode", choices=["export", "finetune", "sensitive", "test"], help="what to run")
    parser.add_argument("--weight", type=str, required=True, help="checkpoint to load (yolov8/yolo11/yolo26)")
    parser.add_argument("--device", type=str, default="cuda:0", help="device")

    # finetune / sensitive / test
    parser.add_argument("--cocodir", type=str, default="/disk1/PeopleCar/v4/",
                        help="dataset root holding images/train and images/val")
    parser.add_argument("--ptq", type=str, default="ptq.pt", help="PTQ checkpoint to save")
    parser.add_argument("--qat", type=str, default="qat.pt", help="QAT checkpoint to save")
    parser.add_argument("--no-eval-origin", dest="eval_origin", action="store_false", help="skip the FP32 baseline eval")
    parser.add_argument("--no-eval-ptq", dest="eval_ptq", action="store_false", help="skip the PTQ eval")
    parser.add_argument("--ignore-policy", type=str, default=None,
                        help="regx of modules to keep in high precision, defaults to the Detect head")
    parser.add_argument("--supervision-stride", type=int, default=1, help="supervision stride")
    parser.add_argument("--iters", type=int, default=3700, help="iters per epoch")
    parser.add_argument("--summary", type=str, default="sensitive-summary.json", help="summary save file")

    # export
    parser.add_argument("--save", type=str, default=None, help="output .onnx path, defaults to beside the weight")
    parser.add_argument("--size", type=int, default=640, help="input size for export onnx")
    parser.add_argument("--output", choices=sorted(OUTPUT_NAMES), default="end2end", help="ONNX output layout")
    parser.add_argument("--no-dynamic", dest="dynamic", action="store_false", help="export a fixed batch size")
    parser.add_argument("--no-simplify", dest="simplify", action="store_false", help="skip onnxsim")
    parser.add_argument("--qadd", action="store_true", help="add QuantAdd to residual connections")
    parser.add_argument("--graphsurgeon", action="store_true", help="run graphsurgeon on the exported onnx")
    parser.add_argument("--ort", action="store_true", help="output=end2end only: ONNX Runtime NMS instead of TensorRT")

    args = parser.parse_args()

    if args.mode == "export":
        run_export(args.weight, args.save, args.size, args.dynamic, not args.qadd,
                   args.output, args.simplify, args.graphsurgeon, args.ort)
    elif args.mode == "finetune":
        run_qat(args.weight, args.cocodir, args.device, args.ignore_policy,
                args.ptq, args.qat, args.supervision_stride, args.iters,
                args.eval_origin, args.eval_ptq)
    elif args.mode == "sensitive":
        run_sensitive_analysis(args.weight, args.device, args.cocodir, args.summary)
    else:
        run_test(args.weight, args.device, args.cocodir)

# ./trtexec --onnx=yolov8m.onnx --saveEngine=yolov8m.engine --minShapes='images':1x3x640x640 --optShapes='images':16x3x640x640 --maxShapes='images':16x3x640x640 --explicitBatch --fp16
# ./trtexec --onnx=qat.onnx --saveEngine=qat.engine --minShapes='images':1x3x640x640 --optShapes='images':16x3x640x640 --maxShapes='images':16x3x640x640 --explicitBatch --fp16
