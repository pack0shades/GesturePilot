import cv2
import cvzone
from cvzone.HandTrackingModule  import HandDetector as htm
import numpy as np
import pandas as pd
import time
#fpsReader = cvzone.FPS(avgCount=10)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 1)

capture_interval = 0.5  # seconds
last_capture_time = time.time()

detector = htm(staticMode=False,modelComplexity=1,detectionCon=0.7,minTrackCon=0.5)

finger_names = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
finger_tip_indexes = [4, 8, 12, 16, 20]

dataset = [] #list to store data

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    current_time = time.time()
    if current_time - last_capture_time < capture_interval:
        # Skip this frame if it is not time to capture
        continue

    last_capture_time = current_time

    hands, img = detector.findHands(img, flipType=False)

    if hands:
        hand1 = hands[0]
        lmList1 = hand1["lmList"]  # List of 21 Landmark points
        bbox1 = hand1["bbox"]  # Bounding box info x,y,w,h
        center1 = hand1['center']  # Center coordinates of the first hand
        handType1 = hand1["type"]  # Type of the first hand ("Left" or "Right")

        fingers1 = detector.fingersUp(hand1)
          
        features = {}
       
        arr = img
        print(f'frame: {arr},',end=',')
        # Print the coordinates of the center of the hand
        center1 = np.array(center1)
        print(f"Center = {center1}", end=',')
        features["center_x"] = center1[0]
        features["center_y"] = center1[1]
        #print which finger is open or close
        for i, finger in enumerate(finger_names):
            
            features[f"{finger}_status"] = fingers1[i]
            print(f"{finger}: {fingers1[i]}",end=',')

        
        #print distance between fingers:
        for k in range(len(finger_tip_indexes)):
            for j in range(k + 1, len(finger_tip_indexes)):
                point1 = lmList1[finger_tip_indexes[k]][0:2]
                point2 = lmList1[finger_tip_indexes[j]][0:2]
                length, info, img= detector.findDistance(point1, point2, img, color=(255, 0, 0), scale=5)
                features[f"Distance{finger_names[k]}{finger_names[j]}"] = length
                print(f"Distance{finger_names[k]}{finger_names[j]}: {length}",end=',')
        print(";")            

        dataset.append(features)
    cv2.imshow("Image", img)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

df = pd.DataFrame(dataset)
df.to_csv("phase1_dataset.csv", index=False)