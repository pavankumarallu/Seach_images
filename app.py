import numpy as np
from PIL import Image
from datetime import datetime
from flask import Flask, request, render_template
from pathlib import Path
import pandas as pd
import numpy as np
import json
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import requests
import numpy as np
import urllib.request
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split as tts
import pickle
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, LeakyReLU, BatchNormalization, Conv2DTranspose
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.models import load_model
from flask_cors import CORS
import numpy as np
from PIL import Image
import os 
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.python.client import device_lib

from sklearn.cluster import KMeans
from skimage.color import rgb2hed
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import metrics
import cv2
import pickle
from tensorflow.keras.models import Sequential
from sklearn.neighbors import NearestNeighbors
import numpy as np
from numpy.linalg import norm
import pickle
import os
import time
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.neighbors import NearestNeighbors

# def extract_features(img_arr, model):
    # expanded_img_array = np.expand_dims(img_arr, axis=0)
    # preprocessed_img = preprocess_input(expanded_img_array)
    # features = model.predict(preprocessed_img)
    # flattened_features = features.flatten()
    # normalized_features = flattened_features / norm(flattened_features)
    # return normalized_features

def extract_features(img_array, model):
    x_train = np.asarray([img_array])
    x_train = x_train.astype('float32') / 255.
    features = model.predict(x_train)
    # features = features.numpy()
    flattened_features = features.flatten()
    normalized_features = flattened_features / norm(flattened_features)
    return normalized_features

def url_to_image(url):
    resp = urllib.request.urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    resized_img = cv2.resize(image,(128,128))
    img = cv2.cvtColor(resized_img,cv2.COLOR_BGR2RGB)
    return img

app = Flask(__name__)
CORS(app, support_credentials=True)

feature_model = tf.keras.models.load_model('Model_final_Dense_128.h5')
model = tf.keras.Model(inputs=feature_model.input,outputs=feature_model.get_layer("Feature_Layer").output)
# model = ResNet50(weights='imagenet', include_top=False,input_shape=(224, 224, 3))
feature_list = pickle.load(open('features_custom.pkl', 'rb'))
indexes = pickle.load(open('index_128.pkl', 'rb'))
neighbors = NearestNeighbors(n_neighbors=15, algorithm='brute',metric='euclidean').fit(feature_list)

with open('Master Sheet_Tiles.json') as json_file:
    data = json.load(json_file)

@app.route('/', methods=['POST'])
def index():
    if request.method == 'POST':
        file = request.get_json(force=True) 
        arr = url_to_image(file['imgSrc'])
        features = extract_features(arr,model)
        distances, indices = neighbors.kneighbors([features])
        ids = indices[0]
        print(ids)
        
        list_img = []
        for i in ids:
            list_img.append(data[indexes[i]])
        
        diction = {"list_img":list_img}
        
        return diction



        

if __name__=="__main__":
    app.run()
    
    app.config['CORS_HEADERS'] = 'Content-Type'
    