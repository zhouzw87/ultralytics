import os
import time
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt
import numpy as np
import cv2


class YOLOXInfer(object):
    def __init__(self, engine_file_path, image_size):
        TRT_LOGGER = trt.Logger(trt.Logger.INFO)
        # 建立模型，创建上下文管理器
        self.engine = trt.Runtime(TRT_LOGGER).deserialize_cuda_engine(open(engine_file_path, 'rb').read())
        self.context = self.engine.create_execution_context()
        self.context.active_optimization_profile = 0

        self.strides = [8, 16, 32]
        self.height = image_size[0]  # 480
        self.width = image_size[1]

        outshape_l = self.context.get_binding_shape(1)
        outshape_m = self.context.get_binding_shape(2)
        outshape_s = self.context.get_binding_shape(3)

        self.context.get_binding_shape(1)
        self.output_l = np.empty(outshape_l, dtype=np.float32)
        self.output_m = np.empty(outshape_m, dtype=np.float32)
        self.output_s = np.empty(outshape_s, dtype=np.float32)

        self.d_input = cuda.mem_alloc(1 * 4 * image_size[0] * image_size[1] * 3)
        self.d_output_l = cuda.mem_alloc(1 * self.output_l.dtype.itemsize * self.output_l.size)
        self.d_output_m = cuda.mem_alloc(1 * self.output_m.dtype.itemsize * self.output_m.size)
        self.d_output_s = cuda.mem_alloc(1 * self.output_s.dtype.itemsize * self.output_s.size)

        self.bindings = [int(self.d_input), int(self.d_output_l), int(self.d_output_m), int(self.d_output_s)]
        self.stream = cuda.Stream()

    def preprocess_input(self, img):
        scale_h = self.height / img.shape[0]
        scale_w = self.width / img.shape[1]
        scale = min([scale_w, scale_h])
        img = cv2.resize(img, None, fx=scale, fy=scale)
        empty_img = np.ones((self.height, self.width, 3), dtype=np.uint8) * 128
        empty_img[0:img.shape[0], 0:img.shape[1], :] = img
        image = empty_img.astype(np.float32)

        image /= 255.0
        image -= np.array([0.485, 0.456, 0.406])
        image /= np.array([0.229, 0.224, 0.225])
        image = np.transpose(image, (2, 0, 1))
        image = np.expand_dims(image, axis=0)
        return image, scale

    def infer(self, img):
        img, scale = self.preprocess_input(img)
        img = np.ascontiguousarray(img, dtype=np.float32)
        cuda.memcpy_htod_async(self.d_input, img, self.stream)
        self.context.execute_async(bindings=self.bindings, stream_handle=self.stream.handle)
        cuda.memcpy_dtoh_async(self.output_s, self.d_output_s, self.stream)
        cuda.memcpy_dtoh_async(self.output_m, self.d_output_m, self.stream)
        cuda.memcpy_dtoh_async(self.output_l, self.d_output_l, self.stream)
        self.stream.synchronize()
        self.output = [self.output_l, self.output_m, self.output_s]

        return self.parse_output(self.output, scale)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def desigmoid(self, y):
        return -np.log(1 / y - 1)

    def parse_output(self, output, scale):
        rects = []
        labels = []
        confs = []
        for k, result in enumerate(output):
            stride = self.strides[k]
            for i in range(result.shape[2]):
                for j in range(result.shape[3]):
                    isObj = self.sigmoid(result[0, 4, i, j])
                    if isObj > 0.5:
                        conf = isObj * self.sigmoid(np.max(result[0, 5:, i, j]))
                        if conf > 0.5:
                            label = np.argmax(result[0, 5:, i, j])

                            x = result[0, 0, i, j]
                            x += j
                            x *= stride
                            y = result[0, 1, i, j]
                            y += i
                            y *= stride
                            w = result[0, 2, i, j]
                            w = np.exp(w)
                            w *= stride
                            h = result[0, 3, i, j]
                            h = np.exp(h)
                            h *= stride

                            x1 = int(max(x - w / 2, 0))
                            y1 = int(max(y - h / 2, 0))
                            x2 = int(max(x + w / 2, 0))
                            y2 = int(max(y + h / 2, 0))
                            rects.append([x1, y1, x2, y2])
                            labels.append(label)
                            confs.append(conf)
        boxes_ids = cv2.dnn.NMSBoxes(rects, confs, 0.2, 0.1)
        objects = []

        for boxes_id in boxes_ids:
            x1 = rects[boxes_id][0] / scale
            y1 = rects[boxes_id][1] / scale
            x2 = rects[boxes_id][2] / scale
            y2 = rects[boxes_id][3] / scale
            label = labels[boxes_id]
            conf = confs[boxes_id]
            obj = {"label": label, "rect": [x1, y1, x2, y2], "conf": conf}
            objects.append(obj)
        return objects


if __name__ == '__main__':
    model = YOLOXInfer('./model/optim_model_fp32.engine', [480, 640])
    image_path = "./dataset/images"
    image_list = os.listdir(image_path)
    for file in image_list:
        filename = os.path.join(image_path, file)
        img = cv2.imread(filename, cv2.IMREAD_COLOR)
        start_time = time.time()
        result = model.infer(img)
        infer_time = time.time() - start_time
        print("time:", infer_time)

        for obj in result:
            x1, y1, x2, y2 = obj["rect"]
            print(obj["rect"])
            ptLeftTop = (int(x1), int(y1))
            ptRightBottom = (int(x2), int(y2))
            point_color = (0, 255, 0)  # BGR
            thickness = 1
            lineType = 4
            cv2.rectangle(img, ptLeftTop, ptRightBottom, point_color, thickness, lineType)
            cv2.putText(img, "FPS:{0:.3f}".format(1 / infer_time), org=(0, 40), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1, color=(0, 255, 0), lineType=2)

            cv2.putText(img, "{}".format(obj["label"]), org=(int(x1), int(y1)), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1, color=(0, 255, 0), lineType=2)
        cv2.imshow('result', img)
        cv2.waitKey(1)
