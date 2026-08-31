from ultralytics import YOLO

# Load a model
# model = YOLO('../ultralytics/cfg/models/v8/yolov8m.yaml').load('pretrains/yolov8m.pt')
# model.train(data='../ultralytics/cfg/datasets/custom_vehicle.yaml', epochs=200, imgsz=640, batch=16,
#             project='vehicle1.4',save_period=1, device=0,
#             degrees=5.0,shear=5.0,perspective=0.0002,mixup=0.2,copy_paste=0.2)

model = YOLO('../trains/vehicle1.4/train2/weights/last.pt')
model.train(resume=True)

# from ultralytics.utils.downloads import download
# from pathlib import Path
#
# # Download labels
# segments = True  # segment or box labels
# dir = Path("../datasets/coco")  # dataset root dir
# url = 'https://github.com/ultralytics/yolov5/releases/download/v1.0/'
# urls = [url + ('coco2017labels-segments.zip' if segments else 'coco2017labels.zip')]  # labels
# download(urls, dir=dir.parent)
# # Download data
# urls = ['http://images.cocodataset.org/zips/train2017.zip',  # 19G, 118k images
#         'http://images.cocodataset.org/zips/val2017.zip',  # 1G, 5k images
#         'http://images.cocodataset.org/zips/test2017.zip']  # 7G, 41k images (optional)
# download(urls, dir=dir / 'images', threads=3)