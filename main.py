'''from PIL import Image
import numpy as np
import os
from random import shuffle
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import numpy as np'''

#-------- serving_sample_request.py ---------#
import base64
import numpy as np
import io
from PIL import Image


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.layers import InputLayer
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from flask import Flask, redirect, render_template
from flask import request
from flask import jsonify
import h5py
#--------------------------------------------------#

app = Flask(__name__)
tf.enable_eager_execution() # Disble graph Tensorflow

@app.route("/")
def hello():
    return "Hello World from Flask Test Model PON"

@app.route("/web", methods=['POST', 'GET'])
def Web():
    #get_model()
    print('Hello Model')
    global model,graph
    model = load_model('modelDogCat.h5')
    #model = load_model('./code/modelDogCat.h5')
    graph = tf.get_default_graph()
    #age = 10
    #return render_template('index.html', data = age)
    return render_template('predict4.html')
#==============================================================#
#==============================================================#

def get_model():
    #---- Load RUN Model (.h5) ---#
    global model,graph
    model = load_model('modelDogCat.h5')
    #model.load_weights('modelDogCat.h5')
    #model._make_predict_function()
    print(" * Model loaded !")


def preprocess_image(image, target_size):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size,Image.ANTIALIAS)
    image = img_to_array(image)
    image = np.expand_dims(image, axis = 0)
    print('preprocess_image: ' + str(image))
    return image


# Evaluate the restored model.
#print("* Loading Model...")
#get_model()
@app.route("/predict", methods=["POST"])
def predict():
    
    message = request.get_json(force=True)
    encoded = message['image']
    print("Hello Pon !!!")
    decoded = base64.b64decode(encoded)
    image = Image.open(io.BytesIO(decoded))
    print("image Open: " + str(image))
    processed_image = preprocess_image(image, target_size=(64,64))
    processed_image = np.float32(processed_image)
    print('processed_image: OK!!')
   
    try:
        model.summary()   
        print('processed_image: --->  ' + str(processed_image))
        # print(model(processed_image))
        prediction = model.predict(processed_image).tolist()
        print('prediction:>> ' + str(prediction))
        d = {0:'cat', 1:'dog'}
        d[prediction[0][0]]
        print(d[prediction[0][0]])
         
        response = {
            'prediction' : {
                #'dog' : prediction[0][1],
                #'cat' : d[prediction[0][0]]
                'result' : d[prediction[0][0]],
                'send' : 'OK!'
             }
        }           
            #return jsonify(prediction)
        return jsonify(response)
    except Exception as e:
        print(e)

if __name__ == "__main__":
    # Only for debugging while developing#
    app.run(host='0.0.0.0',debug=True, port=80)     
    #app.run(debug=True)
