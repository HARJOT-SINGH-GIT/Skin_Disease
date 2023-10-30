# from flask import Flask, render_template, request, jsonify
# from tensorflow.keras.models import load_model
# import numpy as np
# from PIL import Image
# import keras.utils as image
# import tensorflow.keras as keras
# from tensorflow.keras.preprocessing.image import load_img
# from tensorflow.keras.applications.vgg16 import preprocess_input
# import os
# from tensorflow.keras.preprocessing.image import img_to_array
# from tensorflow.keras.preprocessing import image

# app = Flask(__name__)
# model = load_model('sk.h5')

# @app.route('/')
# def index_view():
#     return render_template('index.html')

# def load_image(img):
#     im = Image.open(img)
#     image = np.array(im)
#     return image

# @app.route('/predict', methods=['POST'])
# def predict():
#     if request.method == 'POST':
#         file = request.files['file']
#         if file:
#             img = load_image(file)
#             img = img
#             img = np.resize(img, (28, 28, 3))
    
#             img = image.img_to_array(img)
#             img = np.resize(img, (1, 28, 28, 3))
#             img = img / 255
#             predict = model.predict(img)
#             return jsonify(predict)
#         else: 
#             return "Unable to read the file"

# if __name__ == "__main":
#     app.run(debug=True, host="0.0.0.0", port=8000)


from flask import Flask, render_template , request, jsonify
from tensorflow.keras.models import load_model
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
model=load_model('sk.h5')
# target_img = os.path.join(os.getcwd(),)

@app.route('/')
def index_view():
    return render_template('index.html')

#allow files with extension png,jpg,jpeg
# Allowed_ext


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
            # class_prediction=model.predict(img)
            predict=model.predict(img)
            return jsonify(predict.tolist())
        #     return render_template('predict.html',disease=disease,prob=predict)
        else: 
            return "Unable to read the file"
if __name__ == "__main__":
    app.run(debug=True,host="0.0.0.0",port=process.env.PORT)
    
            
            
            
            # ---------------*----------------
            
# from fastapi import FastAPI, File, UploadFile, Form
# from fastapi.responses import HTMLResponse
# from tensorflow.keras.models import load_model
# from PIL import Image
# import numpy as np
# from starlette.middleware.cors import CORSMiddleware
# from fastapi.templating import Jinja2Templates
# import keras.utils as image
# from tensorflow.keras.preprocessing.image import img_to_array

# app = FastAPI()
# templates = Jinja2Templates(directory="templates")

# model = load_model('sk.h5')

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# def load_image(img):
#     im = Image.open(img.file)
#     image = np.array(im)
#     return image

# @app.get("/", response_class=HTMLResponse)
# async def index_view(request):
#     return templates.TemplateResponse("index.html", {"request": request})

# @app.post("/predict")
# async def predict(file: UploadFile = File(...)):
#     img = load_image(file)
#     img = img
#     img = np.resize(img, (28, 28, 3))
#     img = image.img_to_array(img)
#     img = np.resize(img, (1, 28, 28, 3))
#     # img=img
#     # img=np.resize(img,(28,28,3))
#     img = img / 255
#     predict = model.predict(img)
#     return {"predictions": predict}

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)
