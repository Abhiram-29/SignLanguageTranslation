from ultralytics import  YOLO
import cv2

vid = cv2.VideoCapture(0)
model = YOLO('/home/abhiram/PycharmProjects/ASLDetection/runs/detect/train3/weights/best.pt')

while True:
    ret, frame = vid.read()
    cv2.imshow('frame',frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
vid.release()
cv2.destroyAllWindows()