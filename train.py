import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
import time
import albumentations as A

start_time = time.time()

model = YOLO("yolov8n.yaml")  # build a new model from scratch

# Use the model
model.train(data="config.yaml", epochs=200, resume = True)  # train the model
metrics = model.val()  # evaluate model performance on the validation set
# results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image
path = model.export(format="onnx")  # export the model to ONNX format

end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {execution_time} seconds")