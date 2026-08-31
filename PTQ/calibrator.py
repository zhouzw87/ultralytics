import cv2
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

import os
import numpy as np
from PIL import Image

import torchvision.transforms as transforms


def preprocess_input(image):
    """
    图像预处理
    """
    image = image / 255.0
    # image = image - np.array([0.485, 0.456, 0.406])
    # image = image / np.array([0.229, 0.224, 0.225])

    # image -= np.array([0.5, 0.5, 0.5])
    # image /= np.array([0.5, 0.5, 0.5])
    return image


class YOLOXEntropyCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, args, files_path):
        trt.IInt8EntropyCalibrator2.__init__(self)

        self.cache_file = 'YOLOX.cache'

        self.batch_size = args.batch_size
        self.Channel = args.channel
        self.Height = args.height
        self.Width = args.width
        # self.transform = transforms.Compose([
        #     transforms.Resize([self.Height, self.Width]),  # [h,w]
        #     transforms.ToTensor(),
        # ])

        # 获取数据集中图像的路径列表
        self._txt_file = open(files_path, 'r')
        self._lines = self._txt_file.readlines()
        np.random.shuffle(self._lines)  # ./dataset/images/image_0.jpg;278.35,98.2776,769.969,530.35,1
        self.imgs = [line.strip().split(" ")[0] for line in self._lines]  # 所有文件路径

        # 初始化内存
        self.batch_idx = 0
        self.max_batch_idx = len(self.imgs) // self.batch_size
        self.data_size = trt.volume([self.batch_size, self.Channel, self.Height, self.Width]) * trt.float32.itemsize
        self.device_input = cuda.mem_alloc(self.data_size)

    def next_batch(self):
        """
        读取一个batch的图像数据
        """
        if self.batch_idx < self.max_batch_idx:
            # ***********读取一个batch的文件**************#
            batch_files = self.imgs[self.batch_idx * self.batch_size: \
                                    (self.batch_idx + 1) * self.batch_size]

            batch_imgs = np.zeros((self.batch_size, self.Channel, self.Height, self.Width),
                                  dtype=np.float32)
            for i, f in enumerate(batch_files):
                # img = Image.open(f)
                img = cv2.cvtColor(cv2.imread(f, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
                scale_h = 640 / img.shape[0]
                scale_w = 640 / img.shape[1]
                scale = min([scale_w, scale_h])
                img = cv2.resize(img, None, fx=scale, fy=scale)
                empty_img = np.ones((640, 640, 3), dtype=np.uint8) * 128
                empty_img[0:img.shape[0], 0:img.shape[1], :] = img
                empty_img = empty_img.astype(np.float32)

                img = preprocess_input(empty_img)
                img = img.transpose(2, 0, 1)
                img = img.astype(np.float32)
                img = np.ascontiguousarray(img)

                # 判断字节是否与缓冲区对齐
                assert (img.nbytes == self.data_size / self.batch_size), 'not valid img!' + f
                batch_imgs[i] = img
            self.batch_idx += 1
            print("batch:[{}/{}]".format(self.batch_idx, self.max_batch_idx))
            return np.ascontiguousarray(batch_imgs)
        else:
            return np.array([])

    def get_batch_size(self):
        """
        获取batch大小
        """
        return self.batch_size

    def get_batch(self, names, p_str=None):
        """
        获取一个batch的图像数据，并拷贝到device内存中
        """
        try:
            batch_imgs = self.next_batch()
            if batch_imgs.size == 0 or batch_imgs.size != self.batch_size * self.Channel * self.Height * self.Width:
                return None
            cuda.memcpy_htod(self.device_input, batch_imgs.astype(np.float32))
            return [int(self.device_input)]
        except Exception as e:
            print("发生异常，异常为：{}".format(e))
            return None

    def read_calibration_cache(self):
        """
        读取缓存数据
        """
        # 如果存在校准集的缓存，则使用现有缓存，否则返回空值
        if os.path.exists(self.cache_file):
            print("succeed finding cache file:{}".format(self.cache_file))
            with open(self.cache_file, "rb") as f:
                return f.read()
        else:
            print("failed finding int8 cache!")
            return

    def write_calibration_cache(self, cache):
        """
        写入缓存数据
        """
        with open(self.cache_file, "wb") as f:
            f.write(cache)
        print("succeed saving int8 cache!")