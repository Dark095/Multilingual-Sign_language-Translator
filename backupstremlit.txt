import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model

class HandDetector:
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.maxHands,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon,
        )
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0, draw=True):
        lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        return lmList

# Load the trained TensorFlow model
model = load_model('model_asl/keras_model.h5')

# Load the labels
with open('model_asl/labels.txt', 'r') as f:
    labels = f.read().splitlines()

# Crop and preprocess the hand region
def preprocess_hand_region(frame, lmList):
    x_min = min([lm[0] for lm in lmList])
    y_min = min([lm[1] for lm in lmList])
    x_max = max([lm[0] for lm in lmList])
    y_max = max([lm[1] for lm in lmList])

    # Add some padding around the hand region
    x_min = max(0, x_min - 20)
    y_min = max(0, y_min - 20)
    x_max = min(frame.shape[1], x_max + 20)
    y_max = min(frame.shape[0], y_max + 20)

    hand_region = frame[y_min:y_max, x_min:x_max]
    resized_hand = cv2.resize(hand_region, (224, 224))
    preprocessed_hand = resized_hand / 255.0  # Normalize

    return preprocessed_hand

def main():
    st.title("Multilingual Sign Language Recognizer")
    st.write("Upload a video or use your webcam to recognize sign language gestures.")

    # Select video source
    video_source = st.radio("Choose video source:", ('Webcam', 'Upload'))

    if video_source == 'Webcam':
        run_webcam()
    elif video_source == 'Upload':
        uploaded_video = st.file_uploader("Upload a video file", type=['mp4', 'avi', 'mov'])
        if uploaded_video is not None:
            process_uploaded_video(uploaded_video)

def run_webcam():
    detector = HandDetector(detectionCon=0.7)
    cap = cv2.VideoCapture(0)

    stframe = st.empty()
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.warning("Unable to access webcam. Make sure it's connected.")
            break

        frame = cv2.flip(frame, 1)
        frame = detector.findHands(frame)
        lmList = detector.findPosition(frame, draw=False)

        if len(lmList) != 0:
            preprocessed_hand = preprocess_hand_region(frame, lmList)

            # Add a batch dimension and predict
            prediction = model.predict(np.expand_dims(preprocessed_hand, axis=0))
            class_id = np.argmax(prediction)
            label = labels[class_id]

            cv2.putText(frame, label, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2)

        stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

    cap.release()

def process_uploaded_video(video):
    detector = HandDetector(detectionCon=0.7)
    temp_video_path = "temp_video.mp4"

    with open(temp_video_path, 'wb') as f:
        f.write(video.read())

    cap = cv2.VideoCapture(temp_video_path)

    stframe = st.empty()
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame = detector.findHands(frame)
        lmList = detector.findPosition(frame, draw=False)

        if len(lmList) != 0:
            preprocessed_hand = preprocess_hand_region(frame, lmList)

            # Add a batch dimension and predict
            prediction = model.predict(np.expand_dims(preprocessed_hand, axis=0))
            class_id = np.argmax(prediction)
            label = labels[class_id]

            cv2.putText(frame, label, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2)

        stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

    cap.release()

if __name__ == "__main__":
    main()

# Run the application using: streamlit run Multilingual_sign_language_recognizer.py