import requests
import cv2
url = "http://127.0.0.1:8000/detect/"
files = {'file': open('/home/abhiram/PycharmProjects/ASLDetection/AUG_ASL_Pascal_Voc/test/images/A22_jpg.rf.f02ad8558ce1c88213b4f83c0bc66bc8.jpg', 'rb')}
image = cv2.imread('/home/abhiram/PycharmProjects/ASLDetection/AUG_ASL_Pascal_Voc/test/images/A22_jpg.rf.f02ad8558ce1c88213b4f83c0bc66bc8.jpg')

response = requests.post(url, files=files)

if response.status_code == 200:
    print("Request successful!")
    print(response.text)
else:
    print(f"Request failed with status code {response.status_code}")