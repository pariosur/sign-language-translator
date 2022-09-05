import streamlit as st
import cv2
from PIL import Image

#Display logo on the left and titleon the right
col1, col2 = st.columns([1, 6])
medical_image = Image.open('./medical symbol.png')
with col1:
    st.image(medical_image, width=100)
with col2:
    #st.markdown('<h1 style="float: left;">MEDIAL SERVICES</h1><img style="float: right;" src="MEDICAL SYMBOL.PNG" />', unsafe_allow_html=True)
    st.title("MEDICAL SERVICES")

#Visualize patient info
st.subheader("Patient details")
st.text("Name: XXXXX XXXXX \nAge:XX\nGenre: XXXXX")

#Open camera on the left and display symptoms on the right
col1, col2 = st.columns([1, 1])
with col1:
    run = st.button("Translate patient's symptoms")
    frame_window = st.image([])
    cam = cv2.VideoCapture(0)
    while run:
        ret, frame = cam.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_window.image(frame)
        #with col2:
            #st.text("headache \ntired \nsore throat")
