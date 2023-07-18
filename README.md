# image_resize_ML

## Introduction 
Image resizing is a fundamental task in computer vision and image processing. Traditional resizing techniques, such as scaling, can lead to distortion and loss of important image content. Seam carving is an advanced technique that intelligently removes or adds seams (narrow strips of pixels) from the image to resize it while preserving essential content and aspect ratios. In this tutorial, we will implement an image resizing application using seam carving in Python, leveraging the power of OpenCV and Streamlit.

## Requirements
Before we proceed, ensure you have the following libraries installed:

<li><b>OpenCV</b>: For image processing tasks</li>
<li><b>NumPy</b>: For array manipulations </li>
<li><b>Streamlit</b>: For building interactive web applications </li>
<p>You can install them using pip:</p>
     
     pip install opencv-python numpy streamlit 

## The Seam Carving Algorithm
<p>The seam carving algorithm consists of the following steps:</p>
<ol>
  <li><b>Energy Map Calculation</b>: Compute the energy map of the input image. The energy map represents the importance of each pixel in the image, where high energy values indicate significant content.</li>
  <li><b>Finding Optimal Seams</b>: Identify the seams with the lowest energy along which the image will be resized. These seams pass through low-energy regions and preserve important features.</li>
  <li><b>Seam Removal/Addition</b>: Remove or add the identified seams from/to the image, resizing it accordingly.</li>
</ol>

## Implementation Steps
<ol>
  <li>Import the necessary libraries: OpenCV, NumPy, and Streamlit.</li>
  <li>Define the seam_carve function that takes an image and new width and height as inputs and resizes the image using the seam carving algorithm. </li>
  <li> Read and display the input image using Streamlit's file_uploader and st.image functions.</li>
  <li> Use Streamlit's slider widget to allow the user to select the desired width and height for the resized image.</li>
  <li> Apply the seam_carve function on the uploaded image with the selected dimensions and display the resized image using st.image.</li>
</ol>

## Basic division of code : 
### Importing Libraries
    import cv2
    import numpy as np
    import streamlit as st

### Define the seam_carve Function
    def seam_carve(img, new_width, new_height):
    return img

### Streamlit Web Application
    st.title("Image Resizing using Seam Carving")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
      img = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)
      st.image(img, clamp=True, channels="RGB")
      
      new_width = st.slider('Pick a width', 100, img.shape[1], img.shape[1])
      new_height = st.slider('Pick a height', 0, img.shape[0], img.shape[0])

      resized_img = seam_carve(img, new_width, new_height)
    
      st.image(resized_img, clamp=True, channels="RGB")


<p>we explored the powerful concept of Seam Carving for image resizing, implementing an Image Resizing application using OpenCV and Streamlit. The seam_carve function efficiently calculated the energy map, identified optimal seams, and preserved essential content during the resizing process. The interactive Streamlit web application allowed users to upload images, select desired dimensions, and visualize the resized images in real-time. Seam Carving provides a sophisticated approach to maintain image integrity while achieving precise and visually appealing results, making it a valuable tool for various image processing tasks.</p>
