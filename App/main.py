import numpy as np 
import matplotlib.pyplot as plt 
import os 
import cv2 from keras.applications.resnet50 
import ResNet50 from keras.preprocessing 
import image from keras.applications.resnet50 
import preprocess_input, decode_predictions

model = ResNet50(weights='imagenet')

img_path = 'football.jpg' # The image to classify
#image = cv2.imread('path/to/your/image.jpg')
img = cv2.imread(img_path)

pythonCopy codeimg = cv2.resize(img, (224, 224))  # Resize the image to match the model's input size
x = image.img_to_array(img)  # Convert the image to a numpy array
x = np.expand_dims(x, axis=0)  # Add a batch dimension
x = preprocess_input(x)