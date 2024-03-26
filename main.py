import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
import albumentations as A

model = YOLO("yolov8n.yaml")  # build a new model from scratch

# Use the model
model.train(data="coco128.yaml", epochs=3)  # train the model
# metrics = model.val()  # evaluate model performance on the validation set
# results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image
# path = model.export(format="onnx")  # export the model to ONNX format
