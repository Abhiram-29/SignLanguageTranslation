import requests
import cv2
import json
from io import BytesIO
url = "http://127.0.0.1:8000/detect/"


# files = {'file': open(r'C:\Users\abhir\OneDrive\Pictures\Camera Roll\WIN_20240507_11_25_43_Pro.jpg', 'rb')}
# image = cv2.imread(r'C:\\Users\\abhir\\OneDrive\\Pictures\\Camera Roll\\WIN_20240507_11_25_45_Pro.jpg')

# response = requests.post(url, files=files)
# if response.status_code == 200:
#     print(response.text)
# else:
#     print("Error 500")

# vid = cv2.VideoCapture(0)
# ret,frame = vid.read()
# if ret:
#     # Encode the frame as JPEG
#     _, frame_bytes = cv2.imencode('.jpg', frame)

#     # Create a BytesIO object from the encoded frame
#     frame_file = BytesIO(frame_bytes)

#     # Set the file-like object to the desired position
#     frame_file.seek(0)

#     # Send a POST request to the API endpoint with the frame as a file
#     response = requests.post(url, files={'file': frame_file})

#     # Print the response from the API
#     if response.status_code == 200:
#         print("HELSR")
#     else:
#         print("Error 500")
# else:
#     print("Skipped")
# if response.status_code == 200:
#     print("Request successful!")
#     print(response.text)
# else:
#     print(f"Request failed with status code {response.status_code}")

vid = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX

while True:
    ret, frame = vid.read()
    if ret:
        # Encode the frame as JPEG
        _, frame_bytes = cv2.imencode('.jpg', frame)

        # Create a BytesIO object from the encoded frame
        frame_file = BytesIO(frame_bytes)

        # Set the file-like object to the desired position
        frame_file.seek(0)

        # Send a POST request to the API endpoint with the frame as a file
        response = requests.post(url, files={'file': frame_file})

        # Print the response from the API
        if response.status_code == 200:
            response_json = response.json()
            detections_json = response_json['detections']
            detections_list = json.loads(detections_json)
            # detection = response.text
            print("Detections:",detections_list)
            print("Raw output",response.text)
            if(len(detections_list)):
                cv2.putText(frame, "detected", (10, 450), font, 3, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow('Detected Alphabet', frame)
    else:
        print("Skipped")
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()
