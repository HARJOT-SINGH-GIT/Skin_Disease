from flask import Flask, render_template , request, jsonify
from tensorflow.keras.models import load_model
from flask_cors import CORS
import numpy as np
from PIL import Image
import keras.utils as image
import tensorflow.keras as keras
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications.vgg16 import preprocess_input
import os
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing import image
import os



app= Flask(__name__)
CORS(app)
model=load_model('sk.h5')

@app.route('/')
def index_view():
    return render_template('index.html')




def load_image(img):
    im=Image.open(img)
    image=np.array(im)
    return image

@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
        file=request.files['file']
        if file :
            img = load_image(file)
            img=img
            img=np.resize(img,(28,28,3))
    

            img = image.img_to_array(img)
            img=np.resize(img,(1,28,28,3))
            img = img/255
            
            predict=model.predict(img)
            return jsonify(predict.tolist())
        else: 
            return "Unable to read the file"
if __name__ == "__main__":
<<<<<<< HEAD

=======
>>>>>>> 8886d1f7e833d489f0ded348e8847ef77e1c3f69
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
    
            
