import streamlit as st
import cv2
from PIL import Image

title_container = st.container()
col1, col2 = st.columns([2, 5])
medical_image = Image.open('./medical symbol.png')
with title_container:
    with col1:
        st.image(medical_image, width=100)
    with col2:
        st.markdown('<h1 style="float: left;">MEDIAL SERVICES</h1><img style="float: right;" src="MEDICAL SYMBOL.PNG" />', unsafe_allow_html=True)

#medical_image = Image.open('./medical symbol.png')
#st.image(medical_image, width=100, use_column_width=False)

st.title("Webcam Test")


run = st.button("Translate patient's symptoms:")
if run:
    # print is visible in the server output, not in the page
    print('button clicked!')


frame_window = st.image([])
cam = cv2.VideoCapture(0)

while run:
    ret, frame = cam.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_window.image(frame)
