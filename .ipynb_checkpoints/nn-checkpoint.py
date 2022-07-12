# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten, MaxPooling2D, BatchNormalization
from tensorflow.keras.layers import Conv2D 
import shutil
import numpy as np
import os
import json
import random
import pathlib
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint
from train.bac_resnet import resnet152_model
from PIL import Image
from matplotlib.pyplot import imread
import cv2 as cv
import pandas as pd

#from PIL import ImageDraw, Image
file_dir = "GRAM100/"
batch_size = 256
img_height = img_width = 100
train_image_generator = ImageDataGenerator(rotation_range=30,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest",
    validation_split = 0.25)
'''
for subdir, dirs, files in os.walk("GRAM100"):
        for file in files:
            im = Image.open(os.path.join(subdir, file))
            im = np.expand_dims(im, axis=0)
            imageGen = train_image_generator.flow(im, batch_size=1, save_to_dir=subdir,save_prefix="im", save_format="jpg")
            total = 0
            for image in imageGen:
                total += 1
                if total == 2:
                    break
for subdir, dirs, files in os.walk("GRAM_test100"):
        for file in files:
            im = Image.open(os.path.join(subdir, file))
            im = np.expand_dims(im, axis=0)
            imageGen = train_image_generator.flow(im, batch_size=1, save_to_dir=subdir,save_prefix="im", save_format="jpg")
            total = 0
            for image in imageGen:
                total += 1
                if total == 2:
                    break
'''
train_data_gen = train_image_generator.flow_from_directory(seed = 228, batch_size=batch_size,
                                                           directory=file_dir,
                                                           shuffle=True,
                                                           target_size=(img_height, img_width),
                                                           #save_to_dir = "aug",
                                                           class_mode='categorical', subset='training')
val_data_gen = train_image_generator.flow_from_directory(seed = 228, batch_size=batch_size,
                                                           directory=file_dir,
                                                           shuffle=True,
                                                           target_size=(img_height, img_width),
                                                           #save_to_dir = "aug",
                                                           class_mode='categorical', subset = 'validation')


print(train_data_gen.class_indices)
filepath="weights.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]
num_channels=3
num_classes=len(train_data_gen.class_indices)
'''
model = resnet152_model(img_width, img_height, num_channels, num_classes)
'''

model = Sequential()

model.add(Conv2D(32, (3,3),
                 padding='valid',
                 input_shape=(img_height, img_width, 3),
                 activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Conv2D(32, (3,3),
                 activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Conv2D(16, (3,3),
                 activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Conv2D(16, (3,3),
                 activation='relu'))
model.add(Flatten())
model.add(Dense(units=8, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax', name='fc8'))


weights_path = 'weights.h5'
#model.load_weights(weights_path, by_name=True)
#base_learning_rate = 0.001
model.compile(optimizer=keras.optimizers.Adam(0.001),
              loss='categorical_crossentropy',
              metrics=['categorical_accuracy'])

model.summary()
model.fit(
    train_data_gen,
    steps_per_epoch=train_data_gen.samples // batch_size,
    epochs=10,
    verbose=1,
    callbacks = callbacks_list,
    validation_data=val_data_gen,
    validation_steps=val_data_gen.samples // batch_size
)

test = train_image_generator.flow_from_directory(seed = 228, batch_size=1,
                                                           directory="GRAM_test100",
                                                           shuffle=True,
                                                           target_size=(img_height, img_width),
                                                            class_mode='categorical')
step_test=test.n//test.batch_size  
#(x, y) = test.next()
#print(model.evaluate_generator(test, step_test))
b = pd.read_csv("model2_stat.csv")
l = []
l.append(model.count_params())
for v in (train_data_gen, val_data_gen, test):
    l.append(model.evaluate_generator(v, v.n//v.batch_size)[1])
p = pd.DataFrame([l], columns = ["params", "train", "validate", "test"])
print(p)
b.append(p)
b.to_csv("model2_stat2.csv", sep = ' ', index=False)


'''
fine_tune_at = 710
print(len(model.layers))
for layer in model.layers[:fine_tune_at]:
    layer.trainable =  False
'''