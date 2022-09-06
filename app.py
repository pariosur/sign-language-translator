import streamlit as st
from streamlit_webrtc import VideoProcessorBase, RTCConfiguration,WebRtcMode, webrtc_streamer
#from sign-language-translator import utils as ut
import av
import cv2
import numpy as np
import mediapipe as mp
import os
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.models import load_model

mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    return image, results

def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACE_CONNECTIONS) # Draw face connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS) # Draw pose connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw right hand connections


def draw_styled_landmarks(image, results):
    # Draw face connections
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
                             mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                             mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                             )
    # Draw pose connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                             mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                             )
    # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                             mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                             )
    # Draw right hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                             mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                             )

st.title("ASL SYMPTOM DETECTION")
st.write("Hello, Patient!")


def callback(frame):

    sequence = []
    # sentence = []
    # predictions = []
    # threshold = 0.5

    img = frame.to_ndarray(format="bgr24")

    holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

     # Make detections
    image, results = mediapipe_detection(img, holistic)

    # Draw landmarks
    draw_styled_landmarks(image, results)

    # 2. Prediction logic
    #keypoints = extract_keypoints(results)
    #sequence.append(keypoints)
    # sequence = sequence[-60:]

    return av.VideoFrame.from_ndarray(image, format="bgr24")


webrtc_streamer(key="example", video_frame_callback=callback)









# def sign_language_detector():

#     class OpenCVVideoProcessor(VideoProcessorBase):
#         def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
#             img = frame.to_ndarray(format="bgr24")

#             mp_holistic = mp.solutions.holistic # Holistic model
#             mp_drawing = mp.solutions.drawing_utils # Drawing utilities

#             # Actions that we try to detect
#             actions = np.array(['headache', 'tired', 'sore throat', 'infection'])
#             colors = [(245,117,16), (117,245,16), (16,117,245), (115,11,145)]

#             # Load the model from Modelo folder:

#             model = load_model('action_model.hdf5',actions)

#             # 1. New detection variables
#             sequence = []
#             sentence = []
#             threshold = 0.8

#             # Set mediapipe model
#             with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
#                 while True:
#                     #img = frame.to_ndarray(format="bgr24")
#                     flip_img = cv2.flip(img,1)

#                     # Make detections
#                     image, results = mediapipe_detection(flip_img, holistic)

#                     # Draw landmarks
#                     draw_styled_landmarks(image, results)

#                     # 2. Prediction logic
#                     keypoints = extract_keypoints(results)
#                     sequence.append(keypoints)
#                     sequence = sequence[-30:]

#                     if len(sequence) == 30:
#                         res = model.predict(np.expand_dims(sequence, axis=0))[0]
#                         #print(actions[np.argmax(res)])

#                     #3. Viz logic
#                         if res[np.argmax(res)] > threshold:
#                             if len(sentence) > 0:
#                                 if actions[np.argmax(res)] != sentence[-1]:
#                                     sentence.append(actions[np.argmax(res)])
#                             else:
#                                 sentence.append(actions[np.argmax(res)])

#                         if len(sentence) > 5:
#                             sentence = sentence[-5:]

#                         # Viz probabilities
#                         image = prob_viz(res, actions, image, colors)

#                     cv2.rectangle(image, (0,0), (640, 40), (234, 234, 77), 1)
#                     cv2.putText(image, ' '.join(sentence), (4,30),
#                                 cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
#                     return av.VideoFrame.from_ndarray(image,format="bgr24")

#     webrtc_ctx = webrtc_streamer(
#         key="opencv-filter",
#         mode=WebRtcMode.SENDRECV,
#         video_processor_factory=OpenCVVideoProcessor,
#         async_processing=True,
#     )
