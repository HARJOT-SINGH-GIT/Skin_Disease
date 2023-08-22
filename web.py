import streamlit as st
import tensorflow.keras as keras
from tensorflow.keras.models import load_model
from PIL import Image
import keras.utils as image
import numpy as np
import pandas as pd
import numpy as np 
from tensorflow.keras.preprocessing.image import img_to_array
# load model
model = load_model('\project\skin\sk.h5')

st.write("""# This is the Breast Cancer Clasifier model used to clasify the cancer in the patient""")
select=st.radio("Select the model type to use",["image","textual"],index=0)
if select=="image":
    def load_image(img):
        im=Image.open(img)
        image=np.array(im)
        return image

    upload_file=st.file_uploader("Choose a image file")
    if upload_file is not None:
        img=load_image(upload_file)
        st.image(img)
        st.write("Image uploaded successfully")
    else:
        st.write("Image not uploaded")

    img=img
    img=np.resize(img,(28,28,3))
    # st.write(img.shape)

    img = image.img_to_array(img)
    img=np.resize(img,(1,28,28,3))
    img = img/255


    predict=model.predict(img)
    st.write(predict)
    predict.all()
    # print(predict)
    if predict[0][0]>0.5:
        st.write("The disease is classified as Actinic keratoses and intraepithelial carcinomae (akiec)")
    elif predict[0][1]>0.5:
        st.write("The disease is classified as basal cell carcinoma (bcc)")
    elif predict[0][2]>0.5:
        st.write("The disease is classified as benign keratosis-like lesions (bkl)")
    elif predict[0][3]>0.5:
        st.write("The disease is classified as dermatofibroma (df)")
    elif predict[0][4]>0.5:
        st.write("The disease is classified as melanocytic nevi (nv)")
    elif predict[0][5]>0.5:
        st.write("The disease is classified as pyogenic granulomas and hemorrhage (vasc)")
    elif predict[0][6]>0.5:
        st.write("The disease is classified as melanoma (mel)")
    else:
        st.write("The disease can't be classified")