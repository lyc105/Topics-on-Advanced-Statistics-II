# -*- coding: utf-8 -*-
"""
Created on Tue May 1 15:02:21 2018

@author: Yingkai
"""

import math
import numpy as np
import pandas as pd
import seaborn as sns
from glob import glob
from PIL import Image
import matplotlib.pylab as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split

#from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator, random_rotation, random_shift, random_shear, random_zoom, random_channel_shift, transform_matrix_offset_center, img_to_array

train_images = glob("/home/liyichao/jyk/whale/train/*jpg")
test_images = glob("/home/liyichao/jyk/whale/test/*jpg")
df = pd.read_csv("/home/liyichao/jyk/whale/train.csv")
#train_images = glob("D:/Users/Yingkai/OneDrive/Documents/2018/Deep Learning/project 1/train/*jpg")
#test_images = glob("D:/Users/Yingkai/OneDrive/Documents/2018/Deep Learning/project 1/test/*jpg")
#df = pd.read_csv("D:/Users/Yingkai/OneDrive/Documents/2018/Deep Learning/project 1/train.csv")

SelectedIndex = []
#CountValue = df['Id'].value_counts()
for index, row in df.iterrows():
#    if (CountValue[row[1]] != 1 and row[1] != 'new_whale'):
    if row[1] != 'new_whale':
        SelectedIndex.append(index)
df = df.iloc[SelectedIndex]

df["Image"] = df["Image"].map( lambda x : "/home/liyichao/jyk/whale/train/"+x)
#df["Image"] = df["Image"].map( lambda x : "D:/Users/Yingkai/OneDrive/Documents/2018/Deep Learning/project 1/train/"+x)
selected_train_images = df["Image"]
ImageToLabelDict = dict( zip( df["Image"], df["Id"]))
SIZE = 64
#image are imported with a resizing and a black and white conversion

def Grey(img):
    if img.shape[2] == 3:
        r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
        grey =  0.2989 * r + 0.5870 * g + 0.1140 * b
    else:
        grey = img[:,:,0]
    return grey

def ImportImage( filename):
    img = Image.open(filename).resize( (SIZE,SIZE))
    img = img_to_array(img)
    return Grey(img)

#train_img = []
#y = []
#
#
#for img in selected_train_images:
#    if (CountValue[ImageToLabelDict[img]]< 6):
#        img1 = Image.open(img).resize( (SIZE,SIZE))  
#        img_arr = img_to_array(img1).astype('uint8')
#        RepNum = math.ceil(6/CountValue[ImageToLabelDict[img]])
#        imgs = [random_rotation(img_arr, 50, row_axis=0, col_axis=1, channel_axis=2, fill_mode='nearest')
#        for _ in range(RepNum)]
#        for i in range(RepNum):
#            imgs[i] = random_shift(imgs[i], wrg=0.1, hrg=0.1, row_axis=0, col_axis=1, channel_axis=2, fill_mode='nearest')
#            imgs[i] = random_shear(imgs[i], intensity=0.4, row_axis=0, col_axis=1, channel_axis=2, fill_mode='nearest')
#            imgs[i] = random_zoom(imgs[i], zoom_range=(1.1, 0.8), row_axis=0, col_axis=1, channel_axis=2, fill_mode='nearest')
#        for i in imgs:
#            train_img.append(Grey(i))
#            y.append(ImageToLabelDict[img])   
#        #Image.fromarray(imgs[0])
#        #Image.fromarray(imgs[1])
#        #Image.fromarray(imgs[2])
#        #Image.fromarray(imgs[3])
#        #Image.fromarray(imgs[4])
#    else:
#        train_img.append(ImportImage(img))
#        y.append(ImageToLabelDict[img])      


train_img = np.array([ImportImage( img) for img in selected_train_images])

x = np.array([ImportImage( img) for img in selected_train_images])

class LabelOneHotEncoder():
    def __init__(self):
        self.ohe = OneHotEncoder()
        self.le = LabelEncoder()
    def fit_transform(self, x):
        features = self.le.fit_transform( x)
        return self.ohe.fit_transform( features.reshape(-1,1))
    def transform( self, x):
        return self.ohe.transform( self.la.transform( x.reshape(-1,1)))
    def inverse_tranform( self, x):
        return self.le.inverse_transform( self.ohe.inverse_tranform( x))
    def inverse_labels( self, x):
        return self.le.inverse_transform( x)

y = list(map(ImageToLabelDict.get, selected_train_images))
lohe = LabelOneHotEncoder()
y_cat = lohe.fit_transform(y)
#print(y_cat)
#constructing class weights
WeightFunction = lambda x : 1./x**0.53
ClassLabel2Index = lambda x : lohe.le.inverse_tranform( [[x]])
CountDict = dict( df["Id"].value_counts())
class_weight_dic = { lohe.le.transform( [image_name])[0] : WeightFunction(count) for image_name, count in CountDict.items()}
del CountDict

def plotImages( images_arr, n_images=4):
    fig, axes = plt.subplots(n_images, n_images, figsize=(12,12))
    axes = axes.flatten()
    for img, ax in zip( images_arr, axes):
        if img.ndim != 2:
            img = img.reshape( (SIZE,SIZE))
        ax.imshow( img, cmap="Greys_r")
        ax.set_xticks(())
        ax.set_yticks(())
    plt.tight_layout()
    
#use of an image generator for preprocessing and data augmentation
x = x.reshape( (-1,SIZE,SIZE,1))
input_shape = x[0].shape
x_train = x.astype("float32")
y_train = y_cat

image_gen = ImageDataGenerator(
    #featurewise_center=True,
    #featurewise_std_normalization=True,
    rescale=1./255,
    rotation_range=60,
    width_shift_range=.2,
    height_shift_range=.2,
    horizontal_flip=True)

#training the image preprocessing
image_gen.fit(x_train, augment=True)

#visualization of some images out of the preprocessing
#augmented_images, _ = next( image_gen.flow( x_train, y_train.toarray(), batch_size=8*8))
#plotImages( augmented_images,8)

batch_size = 128 #Original batch_size = 128
num_classes = len(y_cat.toarray()[0])
epochs = 40 #Original epochs = 30
output_times = 5
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')

model = Sequential()
model.add(Conv2D(48, kernel_size=(3, 3),
				 padding='same',
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(48, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(48, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(48, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(48, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(48, (5, 5), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.33))#Original Dropout(0.33)
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.33))#Orinial Dropout(0.33)
model.add(Dense(512, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
model.summary()

import warnings
from os.path import split

models = list()
for output_time in range(output_times):
	model.fit_generator(image_gen.flow(x_train, y_train.toarray(), batch_size=batch_size),
			  steps_per_epoch=  x_train.shape[0]//batch_size,
			  epochs=epochs,
			  verbose=1,
			  class_weight=class_weight_dic)
	model.save("model.v"+str(output_time))
    
    

from keras.models import load_model
model = load_model("model.v4")
x_preds = []
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=DeprecationWarning)
    for img in x_train:
        x = image_gen.standardize(img.reshape(1,SIZE,SIZE))
        y = model.predict_proba(x.reshape(1,SIZE,SIZE,1))
        predicted_args = np.argsort(y)[0][::-1][:100]
        predicted_tags = lohe.inverse_labels( predicted_args)
        x_preds.append(predicted_tags)


y = list(map(ImageToLabelDict.get, selected_train_images))    
acc_vec1 = np.array([(y[t] in x_preds[t][:1]) for t in range(len(y))])
acc_vec3 = np.array([(y[t] in x_preds[t][:3]) for t in range(len(y))])
acc_vec5 = np.array([(y[t] in x_preds[t][:5]) for t in range(len(y))])
acc_vec10 = np.array([(y[t] in x_preds[t][:10]) for t in range(len(y))])
acc_vec50 = np.array([(y[t] in x_preds[t][:50]) for t in range(len(y))])
accs = [np.array([(y[t] in x_preds[t][:s]) for t in range(len(y))]).mean() for s in range(1,101)]

plt.plot(accs)
plt.xlabel('Top-k')
plt.ylabel('Accuracy')
plt.title('Accuracy for top-k prediction')
labels = list(ImageToLabelDict.values())
images = np.concatenate((x_train[labels.index(x_preds[4][1])][np.newaxis,:],
                                 x_train[labels.index(x_preds[4][2])][np.newaxis,:],
                                 x_train[labels.index(x_preds[4][3])][np.newaxis,:],
                                 x_train[labels.index(x_preds[4][4])][np.newaxis,:]),axis=0)
plotImages(images,2)
plt.imshow(x_train[labels.index(x_preds[4][0])].reshape(64,64),cmap="Greys_r")


loss = []
acc = []
logfile = open("log0.log","r+")
lines = logfile.readlines()
for line in lines:
    if line[:5] == "70/70":
        loss.append(line.split("loss: ")[1][:6])
        acc.append(line.split("acc: ")[1][:6])
logfile.close()

plt.subplot(121)  
plt.plot(np.array(loss))
plt.xlabel("Epoch")
plt.title("Loss")
plt.subplot(122)  
plt.plot(np.array(acc))
plt.xlabel("Epoch")
plt.title("Accuracy")



fig = plt.figure()

ax1 = fig.add_subplot(111)
ax1.plot(np.array(loss),'r')
ax1.set_ylabel('Loss')
ax1.set_title("Loss and Accuracy")

ax1.set_xlabel('Epoch')

ax2 = ax1.twinx()  # this is the important function
ax2.plot(np.array(acc))
ax2.set_ylabel('Accuracy')

plt.show()



#	with open("sample_submission.csv","w") as f:
#		with warnings.catch_warnings():
#			f.write("Image,Id\n")
#			warnings.filterwarnings("ignore",category=DeprecationWarning)
#			for image in test_images:
#				img = ImportImage( image)
#				x = img.astype( "float32")
#				#applying preprocessing to test images
#				x = image_gen.standardize( x.reshape(1,SIZE,SIZE))
#				
#				y = model.predict_proba(x.reshape(1,SIZE,SIZE,1))
#				predicted_args = np.argsort(y)[0][::-1][:4]
#				predicted_tags = lohe.inverse_labels( predicted_args)
#				image = split(image)[-1]
#				predicted_tags = "new_whale "+" ".join( predicted_tags)
#				f.write("%s,%s\n" %(image, predicted_tags))

