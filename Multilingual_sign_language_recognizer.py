'''
install requirements.txt and then run the below command.

streamlit run Multilingual_sign_language_recognizer.py


'''

import streamlit as st
import cv2
from HandTrackingModule import HandDetector
from ClassificationModule import Classifier
import numpy as np
import math
import time
from PIL import Image

# Initialize variables
cap = cv2.VideoCapture(0)  # Camera ID == 0
detector1 = HandDetector(maxHands=1)
detector2 = HandDetector(maxHands=2)
offset = 20
imgSize = 300
labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]

classifier1 = Classifier("model_asl/keras_model.h5",  # ASL Model
                         "model_asl/labels.txt")
classifier2 = Classifier("TechableISL_upgrade/keras_model.h5",  # ISL Model
                         "TechableISL_upgrade/labels.txt")

# Streamlit UI
st.title("Multilingual Sign Language Recognizer")

# Sidebar with a small logo
logo = Image.open("logo.ico")  # Load the logo
st.sidebar.image(logo, width=80)  # Set small size with width=80

# Sidebar for model selection
selected_model = st.sidebar.radio("Choose Sign Language Model:", ["American Sign Language", "Indian Sign Language"])
use_code1 = selected_model == "American Sign Language"
use_code2 = selected_model == "Indian Sign Language"

st.sidebar.write(f"Currently using: {selected_model}")

# Add options to view charts
if st.sidebar.button("View ASL Chart"):
    st.sidebar.image("Charts\ASL_CHART.png", use_container_width=True)  # Updated parameter
if st.sidebar.button("View ISL Chart"):
    st.sidebar.image("Charts/ISL_CHART.jpg", use_container_width=True)  # Updated parameter

# Video capture and processing
frame_placeholder = st.empty()

while cap.isOpened():
    success, img = cap.read()
    if not success:
        st.write("Failed to capture video")
        break
    
    imgOutput = img.copy()
    hands, img = (detector1.findHands(img) if use_code1 else detector2.findHands(img))
    try:
        if hands:
            if use_code1:
                hand = hands[0]
                x, y, w, h = hand['bbox']
            else:
                x1, y1, w1, h1 = hands[0]['bbox']
                if len(hands) == 2:
                    x2, y2, w2, h2 = hands[1]['bbox']
                    x, y, w, h = min(x1, x2), min(y1, y2), max(x1 + w1, x2 + w2) - min(x1, x2), max(y1 + h1, y2 + h2) - min(y1, y2)
                else:
                    x, y, w, h = x1, y1, w1, h1
            
            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
            imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]
            aspectRatio = h / w

            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap:wCal + wGap] = imgResize
            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap:hCal + hGap, :] = imgResize

            classifier = classifier1 if use_code1 else classifier2
            prediction, index = classifier.getPrediction(imgWhite, draw=False)

            cv2.rectangle(imgOutput, (x - offset, y - offset - 50), (x - offset + 90, y - offset - 50 + 50),
                          (255, 0, 255), cv2.FILLED)
            cv2.putText(imgOutput, labels[index], (x, y - 26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
            cv2.rectangle(imgOutput, (x - offset, y - offset), (x + w + offset, y + h + offset), (255, 0, 255), 4)
    except:
        pass

    # Convert to RGB and display
    imgRGB = cv2.cvtColor(imgOutput, cv2.COLOR_BGR2RGB)
    frame_placeholder.image(imgRGB, channels="RGB")
