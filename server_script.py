import numpy as np 
import os,sys
from PIL import Image
import matplotlib.pyplot as plot
from optparse import OptionParser

from keras.applications import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from keras import models
from keras import layers
from keras import optimizers

def grayscale(picture):
    res= Image.new(picture.mode, picture.size)
    width, height = picture.size

    for i in range(0, width):
        for j in range(0, height):
            pixel=picture.getpixel((i,j))
            avg=int((pixel[0]+pixel[1]+pixel[2])/3)
            res.putpixel((i,j),(avg,avg,avg))
    res.show()
    return res

def normalize(picture):
    width, height = picture.size
    normalized_array = []
    for j in range(0, height):
	    for i in range(0, width):
		    pixel = picture.getpixel((i,j))
		    normalized_array.append( pixel[0] / 255.0 )
    return np.array(normalized_array)

def loadIsCar():
  model = load_model('/home/ubuntu/car_detection_keras_CNN_model.h5')
  return model

def isCar(imgpath,model):
  row,column = 100,100
  img = Image.open(imgpath)

  img = img.resize((row,column),Image.ANTIALIAS)
  gray_image = grayscale(img)

  X_test = normalize(gray_image)
  X_test = X_test.reshape(1, row, column, 1)  # (1, row, column) 3D input for CNN 

  classes = model.predict(X_test)
  maxVal = classes[0].max()
  indexVal = np.where(classes[0]==maxVal) # result is an array
  
  if (indexVal[0] == 0):
      return 1
  else: 
      return 0

def loadIsDamaged():
  image_size = 150
  #Load the VGG model
  vgg_conv = VGG16(weights='imagenet', include_top=False, input_shape=(image_size, image_size, 3))

  # Freeze the layers except the last 4 layers
  for layer in vgg_conv.layers[:-4]:
      layer.trainable = False

  # Create the model
  model = models.Sequential()
  
  # Add the vgg convolutional base model
  model.add(vgg_conv)
  
  # Add new layers
  model.add(layers.Flatten())
  model.add(layers.Dense(1024, activation='relu'))
  model.add(layers.Dropout(0.5))
  model.add(layers.Dense(2, activation='softmax'))
  
  # Show a summary of the model. Check the number of trainable parameters
  model.summary()

  model.load_weights('/home/ubuntu/dmgornot_weights.h5')
  return model

def isDamaged(imgpath,model):
  image_size = 150 
  img = Image.open(imgpath).resize((image_size,image_size))
  img_arr = np.expand_dims(img_to_array(img), axis=0)

  image = preprocess_input(img_arr)
  prediction = model.predict(image)

  maxval = prediction.max()
  if(maxval == prediction[0][0]):
    return 1
  else:
    return 0

def loadDmgLoc():
  image_size = 150
  #Load the VGG model
  vgg_conv = VGG16(weights='imagenet', include_top=False, input_shape=(image_size, image_size, 3))

  # Freeze the layers except the last 4 layers
  for layer in vgg_conv.layers[:-4]:
      layer.trainable = False

  # Create the model
  model = models.Sequential()
  
  # Add the vgg convolutional base model
  model.add(vgg_conv)
  
  # Add new layers
  model.add(layers.Flatten())
  model.add(layers.Dense(1024, activation='relu'))
  model.add(layers.Dropout(0.5))
  model.add(layers.Dense(3, activation='softmax'))
  
  # Show a summary of the model. Check the number of trainable parameters
  model.summary()

  model.load_weights('/home/ubuntu/dmgloc_weights.h5')
  return model

def dmgLoc(imgpath,model):
  image_size = 150
  img = Image.open(imgpath).resize((image_size,image_size))
  img_arr = np.expand_dims(img_to_array(img), axis=0)

  image = preprocess_input(img_arr)
  prediction = model.predict(image)

  maxval = prediction.max()
  if(maxval == prediction[0][0]):
    return 0
  elif(maxval == prediction[0][1]):
    return 1
  else:
    return 2

def loadDmgSev():
  image_size = 150
  #Load the VGG model
  vgg_conv = VGG16(weights='imagenet', include_top=False, input_shape=(image_size, image_size, 3))

  # Freeze the layers except the last 4 layers
  for layer in vgg_conv.layers[:-4]:
      layer.trainable = False

  # Create the model
  model = models.Sequential()
  
  # Add the vgg convolutional base model
  model.add(vgg_conv)
  
  # Add new layers
  model.add(layers.Flatten())
  model.add(layers.Dense(1024, activation='relu'))
  model.add(layers.Dropout(0.5))
  model.add(layers.Dense(3, activation='softmax'))
  
  # Show a summary of the model. Check the number of trainable parameters
  model.summary()

  model.load_weights('/home/ubuntu/dmgsev_weights.h5')
  return model

def dmgSev(imgpath,model):
  image_size = 150
  img = Image.open(imgpath).resize((image_size,image_size))
  img_arr = np.expand_dims(img_to_array(img), axis=0)

  image = preprocess_input(img_arr)
  prediction = model.predict(image)

  maxval = prediction.max()
  if(maxval == prediction[0][0]):
    return 0
  elif(maxval == prediction[0][1]):
    return 1
  else:
    return 2

model0 = loadIsCar()
model1 = loadIsDamaged()
model2 = loadDmgLoc ()
model3 = loadDmgSev ()

def calclaim (dmg,loc,sev):
  if(dmg==0):
    return 0.5
  cnt = 1
  if(loc == 1 ):
    cnt *= 2
  else:
    cnt *= 3

  if (sev == 0):
    cnt *= 0.5
  elif(sev==1):
    cnt *= 3
  else:
    cnt *=7
  
  return cnt

while(True):
  path = '/home/ubuntu/try/2.jpg'
  car = (isCar(path,model0))
  dmg = (isDamaged(path,model1))
  loc = (dmgLoc(path,model2))
  sev = (dmgSev(path,model3))
  print(car,dmg,loc,sev)
  print(calclaim(dmg,loc,sev))

