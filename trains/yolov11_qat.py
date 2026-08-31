import argparse
from modules import run_qat,run_export,run_sensitive_analysis,run_test


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight', type=str, default='../runs/detect/train/weights/last.pt', help='initial weight patsh')
    parser.add_argument('--cocodir', type=str, default="/media/admin1/eeadd7a5-df7d-49e9-920c-edf0a90b41ed/data/hook/v3.2/" , help="coco directory")
    parser.add_argument("--device", type=str, default="cuda:0", help="device")

    parser.add_argument("--ptq", type=str, default="ptq.pt", help="file")
    parser.add_argument("--qat", type=str, default="qat.pt", help="file")
    parser.add_argument("--eval-origin", default=True,action="store_true", help="do eval for origin model")
    parser.add_argument("--eval-ptq", default=True,action="store_true", help="do eval for ptq model")

    parser.add_argument("--ignore-policy", type=str, default="model\.24\.m\.(.*)", help="regx")
    parser.add_argument("--supervision-stride", type=int, default=1, help="supervision stride")
    parser.add_argument("--iters", type=int, default=3700, help="iters per epoch")
    parser.add_argument("--summary", type=str, default="sensitive-summary.json", help="summary save file")

    parser.add_argument("--confidence", type=float, default=0.001, help="confidence threshold")
    parser.add_argument("--nmsthres", type=float, default=0.45, help="nms threshold")

    parser.add_argument('--exportweight', type=str, default='../runs/detect/peoplecar/weights/last.pt',help='initial weight patsh')
    parser.add_argument('--save', type=str,required=False, help="coco directory")
    parser.add_argument('--size', type=int, default=640, help="input size for export onnx ...")
    parser.add_argument('--dynamic', type=bool, default=True, help="dynamic batch for export onnx ...")
    parser.add_argument("--noqadd", type=bool, default=True, help="export do not add QuantAdd")
    parser.add_argument('--end2end', type=bool,default=True, help='export end2end onnx')
    parser.add_argument('--simplify', type=bool,default=True, help='simplify onnx model')
    parser.add_argument('--graphsurgeon', type=bool, default=False, help='Do graphsurgeon to onnx model ...')
    parser.add_argument('--ort', type=bool,default=False, help='export onnx for onnxruntime')

    parser.add_argument('--export', type=bool, default=False, help='Do Export weight to onnx file ...')
    parser.add_argument('--finetune', type=bool, default=True, help='Do PTQ/QAT finetune ...')
    parser.add_argument('--sensitive', type=bool, default=False, help='Do Sensitive layer analysis ...')
    parser.add_argument('--test', type=bool, default=False, help='Do evaluate ...')

    args = parser.parse_args()

    if args.export:
        run_export(args.exportweight, args.save, args.size, args.dynamic,
                   args.noqadd, args.end2end, args.simplify,args.graphsurgeon,args.ort)
        exit()
    elif args.finetune:
        run_qat(
            args.weight, args.cocodir, args.device, args.ignore_policy,
            args.ptq, args.qat, args.supervision_stride, args.iters,
            args.eval_origin, args.eval_ptq
        )
        exit()
    elif args.sensitive:
        run_sensitive_analysis(args.weight, args.device, args.cocodir, args.summary)
        exit()
    elif args.test:
        run_test(args.weight, args.device, args.cocodir)
        exit()
    else:
        parser.print_help()
 # ./trtexec --onnx=yolov8m.onnx --saveEngine=yolov8m.engine --minShapes='images':1x3x640x640 --optShapes='images':16x3x640x640 --maxShapes='images':16x3x640x640 --explicitBatch --fp16
# ./trtexec --onnx=qat.onnx --saveEngine=qat.engine --minShapes='images':1x3x640x640 --optShapes='images':16x3x640x640 --maxShapes='images':16x3x640x640 --explicitBatch --fp16