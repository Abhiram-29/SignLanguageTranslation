import cv2
from ultralytics import YOLO
import time

start_time = time.time()

model = YOLO("yolov8m-cls.pt")

model.train(data="/home/abhiram/Downloads/archive/ASL_Alphabet_Dataset/", epochs=200)
metrics = model.val()
# path = model.export(format="onnx")  # export the model to ONNX format

end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {execution_time} seconds")