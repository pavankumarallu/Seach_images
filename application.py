import numpy as np
from PIL import Image
from datetime import datetime
from flask import Flask, request, render_template
from pathlib import Path
import pandas as pd
import numpy as np

import tensorflow as tf
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split as tts
import pickle
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, LeakyReLU, BatchNormalization, Conv2DTranspose
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.models import load_model

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

application = Flask(__name__)

# Read image features
# fe = FeatureExtractor()
# features = []
# img_paths = []
# for feature_path in Path("./static/feature").glob("*.npy"):
    # features.append(np.load(feature_path))
    # img_paths.append(Path("./static/img") / (feature_path.stem + ".jpg"))
# features = np.array(features)

new_train = []
open_file = open("Data_new_300.pkl", "rb")
new_train = pickle.load(open_file)
open_file.close()

name = []
open_file = open("names_new_300.pkl", "rb")
name = pickle.load(open_file)
open_file.close()

image_paths = []
for i in name:
    image_paths.append("./static/img_new/"+i)
    
feature_model = tf.keras.models.load_model('Model_final_Dense_new.h5')
train_imggg = np.load("E_train_Flatten_new.npy")
intermediate_layer_model = tf.keras.Model(inputs=feature_model.input,
                                       outputs=feature_model.get_layer("Feature_Layer").output)

knn = NearestNeighbors(n_neighbors=10, algorithm='brute', metric='euclidean')
knn.fit(train_imggg)



@application.route('/', methods=['POST'])
def index():
    if request.method == 'POST':
        file = request.get_json(force=True) 
        print(file)

        # Save query image
        # img = Image.open(file.stream)
        # uploaded_img_path = "static/uploaded/" + datetime.now().isoformat().replace(":", ".") + "_" + file.filename
        # img.save(uploaded_img_path)
# 
        # Run search
        # new_test = []
        # print(str(img))
        # test_img = cv2.imread(uploaded_img_path,cv2.IMREAD_UNCHANGED)
        # test_img = cv2.cvtColor(test_img,cv2.COLOR_BGR2RGB)
        # resized_image = cv2.resize(test_img, (128, 128)) 
        # new_test.append(np.asarray( resized_image, dtype="uint8" ))
        # x_test = np.asarray(new_test)
        # x_test = np.asarray(x_test.astype('float32') / 255.)
        # intermediate_output = intermediate_layer_model(x_test)
        # test_x = intermediate_output.numpy()
        # E_test_flatten = test_x.reshape((-1, np.prod(( 8,8,16))))
        # distances,indeces = knn.kneighbors(E_test_flatten)
        # print(indeces)
        # query = fe.extract(img)
        # dists = np.linalg.norm(features-query, axis=1)  # L2 distances to features
        # ids = indeces[0]
        # dis =  distances[0] 
        # scores = [(dis[id], image_paths[ids[id]]) for id in range(len(ids))]
# 
        # return render_template('index.html',query_path=uploaded_img_path,scores=scores)
    # else:
        # return render_template('index.html')


if __name__=="__main__":
    application.run()