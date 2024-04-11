from ultralytics import  YOLO
import cv2
import time
import supervision as sv    

vid = cv2.VideoCapture(0)
model = YOLO("./runs/detect/train3/weights/best.pt")

prevFrameTime = 0
newFrameTime = 0
font = cv2.FONT_HERSHEY_SIMPLEX

box_annotator = sv.BoundingBoxAnnotator()

while True:
    ret, frame = vid.read()

    # detections = sv.Detections.from_yolov5(results)
    # frame = box_annotator.annotate(scene= frame , detections= detections)
    results = model(frame)
    detections = []
    for result in results:
        boxes = result.boxes
        if len(boxes):
            print("The class is ", boxes.cls.numpy()[0])
            detections.append(int(boxes.cls.numpy()[0]))
    print(detections)

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