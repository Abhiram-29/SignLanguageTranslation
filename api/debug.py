from fastapi import FastAPI
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from fastapi import File, UploadFile

model = YOLO("/runs/detect/nanoTrain/weights/best.pt")
# image = cv2.imread('/home/abhiram/PycharmProjects/ASLDetection/AUG_ASL_Pascal_Voc/test/images/A22_jpg.rf.f02ad8558ce1c88213b4f83c0bc66bc8.jpg')
results = model(
    '/home/abhiram/PycharmProjects/ASLDetection/AUG_ASL_Pascal_Voc/test/images/A22_jpg.rf.f02ad8558ce1c88213b4f83c0bc66bc8.jpg')
for result in results:
    boxes = result.boxes
    boxes = boxes.cpu()
    print("The printed part")
    print(boxes.numpy())
    print("The class is ", boxes.cls.numpy()[0])
    print(type(boxes.cls.numpy()[0]))
