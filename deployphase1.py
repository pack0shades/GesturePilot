import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')
warnings.filterwarnings("ignore", category=UserWarning, module='google.protobuf')

import torch
import torchvision.models as models
import torch.nn as nn
import cv2
import cvzone
from cvzone.HandTrackingModule  import HandDetector as htm
import numpy as np
import pandas as pd
import time
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
from AppOpener import open,close
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import pyautogui as pg

def set_volume_percentage(volume_percentage):
    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(
        IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume = interface.QueryInterface(IAudioEndpointVolume)
    
    # Get the volume range
    min_vol, max_vol, _ = volume.GetVolumeRange()
    
    # Calculate the corresponding dB value for the given percentage
    target_volume_db = min_vol + (volume_percentage / 100.0) * (max_vol - min_vol)
    
    # Set the master volume level
    volume.SetMasterVolumeLevel(target_volume_db, None)

in_features = 17
class classifier(torch.nn.Module):
    def __init__(self,in_features):
        super(classifier,self).__init__()
        self.relu = nn.ReLU()
        #layer 1
        self.layer_1 = nn.Linear(in_features, 32)
        self.batchnorm1 = nn.BatchNorm1d(32)
        self.dropout = nn.Dropout(0.2)

        #layer 2
        self.layer_2 = nn.Linear(32, 32)
        self.batchnorm2 = nn.BatchNorm1d(32)
        self.dropout2 = nn.Dropout(0.2)

        #layer 3

        self.layer_3 = nn.Linear(32, 7)
        self.batchnorm3 = nn.BatchNorm1d(7)
        self.dropout3 = nn.Dropout(0.2)
        
        #softmax

        self.logsoftmax = nn.LogSoftmax(dim=1)


    def forward(self, x):
        x = self.layer_1(x)
        x = self.batchnorm1(x)
        x = self.dropout(x)
        x = self.layer_2(x)
        x = self.batchnorm2(x)
        x = self.relu(x)
        x = self.dropout2(x)
        x = self.layer_3(x)
        x = self.batchnorm3(x)
        x = self.dropout3(x)
        x = self.relu(x)
        output = self.logsoftmax(x)
        return output
    



model_path = 'gesture_classifier.pth'
pred = classifier(in_features)
pred.load_state_dict(torch.load(model_path))

pred.eval()

scaler_path = 'phase1_scaler.pkl'
scaler = joblib.load(scaler_path)

label_encoder_path = 'label_encoderphase1.pkl'
label_encoder = joblib.load(label_encoder_path)

#preprocessing
'''def preprocessing(center,fingers):
    center_x = (center[0] - 640/2) / (640/2)  # Assuming image width is 640
    center_y = (center[1] - 480/2) / (480/2)  # Assuming image height is 480

    # Convert fingers status to binary
    fingers_binary = [1 if finger > 0 else 0 for finger in fingers]

    # Convert to PyTorch tensor
    input_tensor = torch.tensor([center_x, center_y] + fingers_binary, dtype=torch.float32).unsqueeze(0)

    return input_tensor'''


cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 1)

capture_interval = 0  # seconds
last_capture_time = time.time()

detector = htm(staticMode=False,modelComplexity=1,detectionCon=0.7,minTrackCon=0.5)

finger_names = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
finger_tip_indexes = [4, 8, 12, 16, 20]

stable_prediction = None
stable_duration = 0
required_stable_duration = 35  # Number of frames the gesture needs to be stable

global current_cmd 
current_cmd = None
global is_open
is_open = False
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
        features={}

        
        #arr = img
        #print(f'frame: {arr},',end=',')
        # Print the coordinates of the center of the hand
        center1 = np.array(center1)
        #print(f"Center = {center1}", end=',')
        features["center_x"] = center1[0]
        features["center_y"] = center1[1]
        #print which finger is open or close
        for i, finger in enumerate(finger_names):
            
            features[f"{finger}_status"] = fingers1[i]
            #print(f"{finger}: {fingers1[i]}",end=',')

        
        for k in range(len(finger_tip_indexes)):
            for j in range(k + 1, len(finger_tip_indexes)):
                point1 = lmList1[finger_tip_indexes[k]][0:2]
                point2 = lmList1[finger_tip_indexes[j]][0:2]
                length, info, img= detector.findDistance(point1, point2, img, color=(255, 0, 0), scale=5)
                features[f"Distance{finger_names[k]}{finger_names[j]}"] = length
                #print(f"Distance{finger_names[k]}{finger_names[j]}: {length}",end=',')
        gi = list(features.values())
        #print(gi)
        input_tensor = torch.tensor(gi,dtype=torch.float).unsqueeze(0)
        input_tensor_unscaled = torch.tensor(scaler.transform(input_tensor),dtype=torch.float)
        with torch.no_grad():
            out = pred(input_tensor_unscaled)
            #print(f"Model output: {out}")
            predicted_gesture_index = torch.argmax(out, dim=1).item()

        predicted_gesture = label_encoder.inverse_transform([predicted_gesture_index])[0]
        

        if predicted_gesture == 'volume' and current_cmd =='volume':
            vol,_,_ = detector.findDistance(lmList1[4][0:2],lmList1[8][0:2])
            #print(vol,end = '   ')
            if vol>=13 and vol < 113:
                print(vol-13)
                set_volume_percentage(vol-13)
            if vol >= 113:
                set_volume_percentage(100)
            if vol <13:
                set_volume_percentage(0)
        if stable_prediction == predicted_gesture:
            stable_duration +=1
        else:
            stable_prediction = predicted_gesture
            stable_duration = 0

        if stable_duration >= required_stable_duration:

            print(f"Predicted gesture for sample: {predicted_gesture}")
            stable_duration = 0  # Reset the duration counter after a valid prediction
            print("********************************************************")
            if predicted_gesture == 'open' and not is_open:
                current_cmd = 'open'
                open('spotify')
                is_open = True
            elif predicted_gesture == 'close' and is_open:
                current_cmd = 'close'
                close('spotify')
                is_open = False
            if predicted_gesture == 'play' and current_cmd != 'play':
                current_cmd = 'play'
                pg.press('playpause')
            if predicted_gesture == 'pause' and current_cmd != 'pause':
                current_cmd = 'pause'
                pg.press('playpause')
            if predicted_gesture == 'next':
                current_cmd = 'next'
                pg.press('nexttrack')
            if predicted_gesture == 'previous':
                current_cmd = 'previous'
                pg.press('prevtrack',presses = 2,interval=1)
            if predicted_gesture == 'close' and current_cmd == close:
                pg.press('q')
            if predicted_gesture == 'volume' and current_cmd!='volume':
                current_cmd = 'volume'    
        
    cv2.imshow("Image", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break