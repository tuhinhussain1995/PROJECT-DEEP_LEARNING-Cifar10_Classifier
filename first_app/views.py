from django.shortcuts import render
from django.http import HttpResponse
from django.contrib import messages


def index(request):
    return render(request, 'index.html')



import numpy as np
import cv2
import os
import pywt
import matplotlib
from matplotlib import pyplot as plt
import joblib
import pickle
import json

import base64
import cv2


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))




########################### This is the most important part to load Deep Learning Network #############################

import tensorflow as tf
config = tf.compat.v1.ConfigProto(gpu_options =
                         tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8)
# device_count = {'GPU': 1}
)
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)



################################### Load the Model #############################################

modeldir = os.path.join(BASE_DIR, 'static/ml_files/cifar10_model.h5')


import tensorflow as tf
from tensorflow.keras.models import load_model
loaded_model = load_model(modeldir)


#################################### Load the CV2 ##################################################

import cv2
import matplotlib.pyplot as plt




from django.core.files.storage import FileSystemStorage
def uploadPic(request):
    if request.method == 'POST':
        uploaded_file = request.FILES['image']
        fs = FileSystemStorage()
        name = fs.save(uploaded_file.name, uploaded_file)

        imgdir = os.path.join(BASE_DIR, 'media', name)

        # imagedir = os.path.join(BASE_DIR, 'static/images/cifar10_images/airplane.jpg')

        img = cv2.imread(imgdir)

        scaled_img = cv2.resize(img, (32, 32))

        print(scaled_img.shape)

        prediction = loaded_model.predict_classes(scaled_img.reshape(1, 32, 32, 3))


        winner = ''

        if prediction[0] == 0:
            winner = 'AIRPLANE'
        elif prediction[0] == 1:
            winner = 'AUTOMOBILE'
        elif prediction[0] == 2:
            winner = 'BIRD'
        elif prediction[0] == 3:
            winner = 'CAT'
        elif prediction[0] == 4:
            winner = 'DEER'
        elif prediction[0] == 5:
            winner = 'DOG'
        elif prediction[0] == 6:
            winner = 'FROG'
        elif prediction[0] == 7:
            winner = 'HORSE'
        elif prediction[0] == 8:
            winner = 'SHIP'
        elif prediction[0] == 9:
            winner = 'TRUCK'


        x = imgdir
        image_path = []

        for char in x:
            if char != '\\':
                image_path.append(char)
            else:
                image_path.append('/')

        final_img_path = ''.join(image_path)
        print(final_img_path)



        return render(request, 'index.html', {'winner' : winner, 'final_img_path' : final_img_path})
