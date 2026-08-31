import tensorrt as trt
import argparse
import calibrator

# 设置模式
EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)


def ONNX2TRT(args, calib=None):
    '''
    :brief:  convert onnx to tensorrt engine, use mode of ['fp32', 'fp16', 'int8']
    :return: trt engine
    '''

    # 判断模式是否可用
    assert args.mode.lower() in ['fp32', 'fp16', 'int8'], "mode should be in ['fp32', 'fp16', 'int8']"

    G_LOGGER = trt.Logger(trt.Logger.WARNING)
    with trt.Builder(G_LOGGER) as builder, \
            builder.create_network(EXPLICIT_BATCH) as network, \
            trt.OnnxParser(network, G_LOGGER) as parser, \
            builder.create_builder_config() as config, \
            trt.Runtime(G_LOGGER) as runtime:

        # 配置batch size
        # builder.max_batch_size = args.batch_size

        profile = builder.create_optimization_profile()
        profile.set_shape('images',(1,3,640,640),(8,3,640,640),(16,3,640,640))#最小的尺寸,常用的尺寸,最大的尺寸,推理时候输入需要在这个范围内
        config.add_optimization_profile(profile)

        # 配置tensorrt的推理缓冲区大小，即构建阶段可用的显存大小
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4096 * (1 << 30))
        # config.max_workspace_size = 1 << 50

        if args.mode.lower() == 'int8':
            assert (builder.platform_has_fast_int8 == True), "not support int8"
            # 配置int8量化所需的参数及校准集
            config.set_flag(trt.BuilderFlag.INT8)
            config.int8_calibrator = calib

        elif args.mode.lower() == 'fp16':
            assert (builder.platform_has_fast_fp16 == True), "not support fp16"
            # 配置fp16模式下的参数
            config.set_flag(trt.BuilderFlag.FP16)
            # config.fp16

        # 加载onnx模型，并解析
        print('Loading ONNX file from path {}...'.format(args.onnx_file_path))
        with open(args.onnx_file_path, 'rb') as model:
            print('Beginning ONNX file parsing')
            if not parser.parse(model.read()):  # parser是tensorrt的onnx解析类，声明位置见20行
                print("ERROR: Failed to parse the ONNX file.")
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                return None
        print(network.get_input(0).shape)
        # network.get_input(0).shape = [1, 3, 640, 960]
        print('Completed parsing of ONNX file')

        # 构建序列化引擎文件
        print('Building an engine from file {}; this may take a while...'.format(args.onnx_file_path))
        # 根据配置及解析的网络构建序列化会话
        plan = builder.build_serialized_network(network, config)
        # 反序列化加载构建推理引擎
        engine = runtime.deserialize_cuda_engine(plan)
        print("Completed creating Engine")
        with open(args.engine_file_path, "wb") as f:
            f.write(plan)
        return engine


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pytorch2TensorRT args")
    parser.add_argument("--batch_size", type=int, default=128, help='batch_size')
    parser.add_argument("--channel", type=int, default=3, help='input channel')
    parser.add_argument("--height", type=int, default=640, help='input height')
    parser.add_argument("--width", type=int, default=640, help='input width')
    parser.add_argument("--cache_file", type=str, default='YOLOX.cache', help='cache_file')
    parser.add_argument("--mode", type=str, default='int8', help='fp32, fp16 or int8')
    parser.add_argument("--onnx_file_path", type=str, default='../trains/yolov8m.onnx', help='onnx_file_path')
    parser.add_argument("--engine_file_path", type=str, default='optim_model_int8.engine',
                        help='engine_file_path')
    args = parser.parse_args()
    calib = calibrator.YOLOXEntropyCalibrator(args,'../datasets/cocotest/val2007.txt')
    ONNX2TRT(args, calib)