import streamlit as st
from ultralytics import YOLO
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import cv2


model =YOLO('last.pt')



st.title('Pisces encyclopedia')
uploaded_file = st.file_uploader("Choose a file")
url = st.text_input('Entrer un url')
button1 = st.button("Prédire")


    
    
   


if button1:

    if uploaded_file :
        img= uploaded_file.getvalue()
        img = np.asarray(img)
        img =Image.open(BytesIO(img))
        res = model.predict(img)
        res_plotted = res[0].plot()
        
        cv2.imwrite("result.png", res_plotted)
        st.image(Image.open('result.png'))
    elif url :
        response = requests.get(url)
        image = Image.open(BytesIO(response.content))
        image = np.asarray(image)
        res = model.predict(image)
        res_plotted = res[0].plot(img=image)
        cv2.imwrite("result.png", res_plotted)
        st.image(Image.open('result.png'))



