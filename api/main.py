from fastapi import FastAPI
import cv2
import numpy as np
from ultralytics import YOLO
from fastapi import File, UploadFile
import torch
import json

model = YOLO("/home/abhiram/PycharmProjects/ASLDetection/runs/detect/train3/weights/best.pt")
app = FastAPI()


@app.post("/detect/")
async def detect_objects(file: UploadFile):
    # Process the uploaded image for object detection
    image_bytes = await file.read()
    image = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    results = model(image)
    detections = []
    for result in results:
        boxes = result.boxes
        boxes = boxes.cpu()
        print("The printed part")
        print(boxes.numpy())
        print("The class is ", boxes.cls.numpy()[0])
        detections.append(int(boxes.cls.numpy()[0]))
    print(detections)
    detections = json.dumps(detections)
    return {"detections": detections}