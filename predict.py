from ultralytics import  YOLO
import cv2

model = YOLO("/runs/detect/nanoTrain/weights/last.pt")
source = "/home/abhiram/PycharmProjects/ASLDetection/AUG_ASL_Pascal_Voc/test/images"
model.predict(source)
