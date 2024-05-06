from ultralytics import  YOLO
import cv2
import time

vid = cv2.VideoCapture(0)
model = YOLO('/runs/detect/nanoTrain/weights/best.pt')

prevFrameTime = 0
newFrameTime = 0
font = cv2.FONT_HERSHEY_SIMPLEX

while True:
    ret, frame = vid.read()
    newFrameTime = time.time()
    fps = 1 / (newFrameTime - prevFrameTime)
    prevFrameTime = newFrameTime
    fps = str(int(fps))
    cv2.putText(frame, fps, (7, 70), font, 3, (100, 255, 0), 3, cv2.LINE_AA)
    cv2.imshow('frame',frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
vid.release()
cv2.destroyAllWindows()