import cv2
import numpy as np
import streamlit as st


st.title("Seam Carving Image Resizing")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])


if uploaded_file is not None:
    img = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8),1)