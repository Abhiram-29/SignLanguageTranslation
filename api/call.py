import requests
import cv2
url = "http://127.0.0.1:8000/detect/"
# files = {'file': open('/home/abhiram/PycharmProjects/ASLDetection/AUG_ASL_Pascal_Voc/test/C/C17_jpg.rf.ceb81f8ae3c3673bd060ebe71848eca8.jpg', 'rb')}
# image = cv2.imread('/home/abhiram/PycharmProjects/ASLDetection/AUG_ASL_Pascal_Voc/test/C/C17_jpg.rf.ceb81f8ae3c3673bd060ebe71848eca8.jpg')

# response = requests.post(url, files=files)
#
# if response.status_code == 200:
#     print("Request successful!")
#     print(response.text)
# else:
#     print(f"Request failed with status code {response.status_code}")

vid = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX

while True:
    ret, frame = vid.read()
    response = requests.post(url, files=frame)
    if response.status_code == 200:
        detection = response.text["detections"]
        cv2.putText(frame, detection, (10, 450), font, 3, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('Detected Alphabet', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()
