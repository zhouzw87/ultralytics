import os
import glob
import cv2
import pycuda.driver as cuda
import tensorrt as trt
import numpy as np
from PIL import Image
import time
import random
import pycuda.autoinit

random.seed(0)
CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
           'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
           'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
           'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
           'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
           'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
           'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
           'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
           'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
           'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
           'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
           'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
           'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
           'scissors', 'teddy bear', 'hair drier', 'toothbrush')

# colors for per classes
COLORS = {
    cls: [random.randint(0, 255) for _ in range(3)]
    for i, cls in enumerate(CLASSES)
}

BATCH_SIZE = 1
INPUT_SHAPE_W_BS = (BATCH_SIZE, 3, 640, 640)
ALLOWED_EXTENSIONS = (".jpeg", ".jpg", ".png")
img_path = "../datasets/coco128/images/val2017"

src_files = [
    path for path in glob.iglob(os.path.join(img_path, "**"), recursive=True)
    if os.path.isfile(path) and path.lower().endswith(ALLOWED_EXTENSIONS)
]
if len(src_files) == 0:
    raise Exception(
        "ERROR: src data path [{}] contains no files!".format(img_path))

    # Add files for making a multiple of batch size
    if len(src_files) % BATCH_SIZE != 0:
        src_files += src_files[len(src_files) % BATCH_SIZE: BATCH_SIZE]
    # initialize batch
init_batch = np.zeros(INPUT_SHAPE_W_BS, dtype=np.float32)
# make batch
init_imgs = []


def preprocess(input_image):
    """
    Preprocesses the input image before performing inference.

    Returns:
        image_data: Preprocessed image data ready for inference.
    """
    # Read the input image using OpenCV
    # img = cv2.imread(input_image)

    # Get the height and width of the input image
    # img_height, img_width = input_image.shape[:2]

    # Convert the image color space from BGR to RGB
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

    # Resize the image to match the input shape
    input_image = cv2.resize(input_image, (640, 640))

    # Normalize the image data by dividing it by 255.0
    image_data = np.array(input_image) / 255.0

    # Transpose the image to have the channel dimension as the first dimension
    image_data = np.transpose(image_data, (2, 0, 1))  # Channel first

    # Expand the dimensions of the image data to match the expected input shape
    image_data = image_data.astype(np.float32).ravel()

    # Return the preprocessed image data
    return image_data

def draw_detections(img, box, score, class_id):
    """
    Draws bounding boxes and labels on the input image based on the detected objects.

    Args:
        img: The input image to draw detections on.
        box: Detected bounding box.
        score: Corresponding detection score.
        class_id: Class ID for the detected object.

    Returns:
        None
    """

    # Extract the coordinates of the bounding box
    x1, y1, w, h = box

    # Retrieve the color for the class ID
    # color = COLORS[class_id]
    color = (0,255,255)
    # Draw the bounding box on the image
    cv2.rectangle(img, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color, 2)

    # Create the label text with class name and score
    label = f"{CLASSES[class_id]}: {score:.2f}"

    # Calculate the dimensions of the label text
    (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

    # Calculate the position of the label text
    label_x = x1
    label_y = y1 - 10 if y1 - 10 > label_height else y1 + 10

    # Draw a filled rectangle as the background for the label text
    cv2.rectangle(
        img, (label_x, label_y - label_height), (label_x + label_width, label_y + label_height), color, cv2.FILLED
    )

    # Draw the label text on the image
    cv2.putText(img, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

def postprocess(input_image, output, confidence_thres=0.6, iou_thres=0.45):
    """
    Performs post-processing on the model's output to extract bounding boxes, scores, and class IDs.

    Args:
        input_image (numpy.ndarray): The input image.
        output (numpy.ndarray): The output of the model.

    Returns:
        numpy.ndarray: The input image with detections drawn on it.
    """

    # Transpose and squeeze the output to match the expected shape
    output_ = output[0].reshape((1,84,8400))
    outputs = np.transpose(np.squeeze(output_))

    # outputs = outputs.reshape((84,8400))
    # Get the number of rows in the outputs array
    rows = outputs.shape[0]

    # Lists to store the bounding boxes, scores, and class IDs of the detections
    boxes = []
    scores = []
    class_ids = []

    # img = cv2.imread(input_image)
    # Get the height and width of the input image
    img_height, img_width = input_image.shape[:2]
    # Calculate the scaling factors for the bounding box coordinates
    x_factor = img_width / 640
    y_factor = img_height / 640

    # Iterate over each row in the outputs array
    for i in range(rows):
        outputs_i = outputs[i]
        # print(outputs_i)
        # Extract the class scores from the current row
        classes_scores = outputs[i][4:]

        # Find the maximum score among the class scores
        max_score = np.amax(classes_scores)

        # If the maximum score is above the confidence threshold
        if max_score > confidence_thres:
            # Get the class ID with the highest score
            class_id = np.argmax(classes_scores)

            # Extract the bounding box coordinates from the current row
            x, y, w, h = outputs[i][0], outputs[i][1], outputs[i][2], outputs[i][3]

            # Calculate the scaled coordinates of the bounding box
            left = int((x - w / 2) * x_factor)
            top = int((y - h / 2) * y_factor)
            width = int(w * x_factor)
            height = int(h * y_factor)

            # Add the class ID, score, and box coordinates to the respective lists
            class_ids.append(class_id)
            scores.append(max_score)
            boxes.append([left, top, width, height])

    # Apply non-maximum suppression to filter out overlapping bounding boxes
    indices = cv2.dnn.NMSBoxes(boxes, scores, confidence_thres, iou_thres)

    # Iterate over the selected indices after non-maximum suppression
    for i in indices:
        # Get the box, score, and class ID corresponding to the index
        box = boxes[i]
        score = scores[i]
        class_id = class_ids[i]

        # Draw the detection on the input image
        draw_detections(input_image, box, score, class_id)

    # Return the modified input image
    return input_image

class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        """
        Within this context, host_mom means the cpu memory and device means the GPU memory
        """
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

def allocate_buffers(engine, batch_size=1):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(-size if size < 0 else size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream

def do_inference(context, bindings, inputs, outputs, stream, batch_size=1):
    # Transfer data from CPU to the GPU.
    context.set_binding_shape(0, [batch_size, *context.get_binding_shape(0)[1:]])
    for inp in inputs:
        device_ptr = inp.device  # binding array
        host_array = inp.host  # input array
        cuda.memcpy_htod_async(device_ptr, host_array, stream)

        # Run inference.
        # context.execute_v2(bindings=bindings)  # for Profiling
        context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)  # for Inferencing
        # Transfer predictions back from the GPU.
        for out in outputs:
            cuda.memcpy_dtoh_async(out.host, out.device, stream)

        # Synchronize the stream
        stream.synchronize()

        # Return only the host outputs.
        host_outputs = []
        for out in outputs:
            host_outputs.append(out.host)
        return host_outputs

if __name__ == '__main__':
    # with open("../trains/qat.engine", 'rb') as f, trt.Runtime(trt.Logger(trt.Logger.WARNING)) as runtime:
    #     engine = runtime.deserialize_cuda_engine(f.read())
    #     inputs, outputs, bindings, stream = allocate_buffers(engine, batch_size=BATCH_SIZE)
    #     context = engine.create_execution_context()
    logger = trt.Logger(trt.Logger.INFO)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(open("../trains/qat_new/ptq.engine", 'rb') .read())
    inputs, outputs, bindings, stream = allocate_buffers(engine)
    outputshape = [engine.get_binding_shape(binding) for binding in engine][1]
    context = engine.create_execution_context()

    infer_times = 0.0
    count=0
    for i in range((len(src_files))):
        count +=1
        # batch,imgs = get_batch(batches)
        print(src_files[i])
        img = cv2.imread(src_files[i])
        img_data = preprocess(img)
        inputs[0].host = img_data
        init_time = time.time()
        trt_output = do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream, batch_size=BATCH_SIZE)
        infer_time = (time.time()-init_time)*1000
        infer_time = 0 if infer_time>10.0 else infer_time
        print(infer_time)
        infer_times += infer_time
        postprocess(img,trt_output,0.5,0.65)
        # output_shape = []
        #
        # # for removing order dependency of outputs, use binding index of engine and get name and shape
        # for binding_idx in range(1, 4): # index 0 is input, index 1 ~ 4 are outputs
        #     output_shape.append({"name": engine.get_binding_name(binding_idx), \
        #                          "shape": (BATCH_SIZE, *engine.get_binding_shape(binding_idx)[1:])})
        save_file = "../result/"+src_files[i].split("/")[-1]
        # print(save_file)
        cv2.imwrite(str(save_file), img)
    print("av_time:",infer_times/count)
