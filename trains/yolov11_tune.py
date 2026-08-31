from ultralytics import YOLO
from ray import tune
# Load a YOLOv8n model
print(tune.is_session_enabled)

model = YOLO('../ultralytics/cfg/models/v8/yolov11m.yaml').load('pretrains/yolov11m.pt')

search_space = {
    'lr0': tune.loguniform(1e-4, 1e-2),
    'momentum': tune.uniform(0.8, 0.95),
    'weight_decay': tune.loguniform(1e-4, 1e-2)
}

# Start tuning hyperparameters for YOLOv8n training on the COCO8 dataset
result_grid = model.tune(data='/home/admin1/ultralytics/ultralytics/cfg/datasets/coco128.yaml',
                         iterations=2,
                         gpu_per_trial=1,
                         space={"lr0": tune.uniform(1e-5, 1e-1)},
                         epochs=10,
                         use_ray=True
                         )

for i, result in enumerate(result_grid):
    print(f"Trial #{i}: Configuration: {result.config}, Last Reported Metrics: {result.metrics}")

import matplotlib.pyplot as plt
for result in result_grid:
    plt.plot(result.metrics_dataframe["training_iteration"], result.metrics_dataframe["mean_accuracy"], label=f"Trial {i}")
plt.xlabel('Training Iterations')
plt.ylabel('Mean Accuracy')
plt.legend()
plt.show()