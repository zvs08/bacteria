# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
#from tensorflow import keras
import keras
#import sklearn
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten, MaxPooling2D, BatchNormalization
from tensorflow.keras.layers import Conv2D 
import shutil
import numpy as np
import numpy.ma as ma
import os
import json
import random
import pathlib
import matplotlib.pyplot as plt
#from tensorflow.keras.callbacks import ModelCheckpoint
from keras.callbacks import ModelCheckpoint
from train.bac_resnet import resnet152_model
from PIL import Image
from matplotlib.pyplot import imread
import cv2
import pandas as pd
#from tensorflow.keras import backend as K
from keras import backend as K
from keras.utils import np_utils

#from PIL import ImageDraw, Image
batch_size = 32
img_height = img_width = 100

def pred_noise(model, im_path):
    temp_dir = "temp" + str(model.count_params())
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir, ignore_errors=True)
    if not os.path.exists(temp_dir):
        os.mkdir(temp_dir)
    save_dir = im_path[im_path.find("/") + 1:im_path.find(".")]
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir, ignore_errors=True)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    if not os.path.exists(save_dir + "/bacterias"):
        os.mkdir(save_dir + "/bacterias")
    if not os.path.exists(save_dir + "/noise"):
        os.mkdir(save_dir + "/noise")
    img = cv2.imread(im_path)
    k = 0
    while(1):
        a = img[k, :, :]
        if a.mean() > 8:
            break
        k += 1
    img2 = img[k:, :, :]
    #plt.imshow(img)

    for i in range(0, 3):
        tresh = np.quantile(img2[:, :, i], 0.08)
        r = ma.masked_greater_equal(img2[:, :, i], tresh)
        img2[:, :, i] = r.filled(255)
    #cv2.imwrite(save_dir + "/img2.jpg", img2)
    x_gr = 100
    y_gr = 100
    pred = []
    for x_off in range(0, 25, 5):
        for y_off in range(0, 25, 5):
            for i in range(0, int((img2.shape[0] - y_off)/ y_gr)):
                for j in range(0, int((img2.shape[1] - x_off)/ x_gr)):
                    img_p = img2[y_off + i*y_gr:y_off + (i+1)*y_gr, x_off + j*x_gr:x_off + (j+1)*x_gr, :]
                    if (255 - img_p).sum() > 1500:
                        #print("i = " + str(i) + ", j = " + str(j) + ", x_off = " + str(x_off) + ", y_off = " + str(y_off))
                        if x_off == 0 and y_off == 0 and i == 0 and j == 0:
                            save_name = "x_off_" + str(x_off) + "_y_off_" + str(y_off) + "_" + str(i) + '_' + str(j) + ".jpg"
                        pred.append(img_p)
                        cv2.imwrite(temp_dir + "/" + "x_off_" + str(x_off) + "_y_off_" + str(y_off) + "_" + str(i) + '_' + str(j) + ".jpg", img_p)
            
    im_gen = ImageDataGenerator(rescale = 1./255)

    c = 0.7
    for subdir, dirs, files in os.walk(temp_dir):
        for file in files:
            im = Image.open(os.path.join(subdir, file))

            im = np.expand_dims(im, axis=0)
            x = im_gen.flow(im, batch_size=1)
            y = x.next()
            p = model.predict(y)
            if np.argmax(p) != 5 and p[0][np.argmax(p)] >= c:
                shutil.copy2(temp_dir + "/" + file, save_dir + "/bacterias/" + file)
            else:
                shutil.copy2(temp_dir + "/" + file, save_dir + "/noise/" + file)
    shutil.rmtree(temp_dir, ignore_errors=True)
def pred_on_image(model, im_path):
    save_dir = im_path[im_path.find("/") + 1:im_path.find(".")] + "/bacterias"
    img_height = img_weight = 100
    
    im_gen = ImageDataGenerator(rescale = 1./255)

    l = []
    dic = {}
    for i in range(0, 5):
        dic[i] = 0

    c = 0.5
    for subdir, dirs, files in os.walk(save_dir):
        for file in files:
            im = Image.open(os.path.join(subdir, file))
            '''
            z = np.array(im)
            
            #print(z.shape)
            for i in range(10, z.shape[2]):
                print("z" + str(i) + " = " + str(z[:, :, i]))
            '''
            im = np.expand_dims(im, axis=0)
            x = im_gen.flow(im, batch_size=1)
            
            y = x.next()
            p = model.predict(y)
            if p[0][np.argmax(p)] >= c:
                lst = [file]
                for k in range(0, p.shape[1]):
                    lst.append(p[0][k])
                lst.append(np.argmax(p))
                    #print("lst = " + str(lst))

                l.append(lst)
                dic[np.argmax(p)] += 1
    '''
    df = pd.DataFrame(l, columns = ["file", 1, 2, 3, 4, 5, "pred"])
    df.to_csv("pred_" + str(model.count_params()) + im_path[im_path.find("/") + 1:im_path.find(".")] + ".csv", index=False)
    #print(df)
    '''
    #shutil.rmtree(save_dir, ignore_errors=True)
    print(dic)
    print("predict = " + str(np.argmax(dic.values())))
    print("-------------------------------")   
    return np.argmax(dic.values())


train_image_generator = ImageDataGenerator(rotation_range=90,vertical_flip = True, horizontal_flip=True, zoom_range=0.0, fill_mode="nearest")
'''
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
'''
''' 
file_dir = "GRAM_red_100_train"

if not os.path.exists("train"):
    os.mkdir("train")
if not os.path.exists("test"):
    os.mkdir("test")
    
for l in ["train"]: 
    if l == "train":
        file_dir = "GRAM_red_100"
    else:
        file_dir = "GRAM_red_100_test"
    for subdir, dirs, files in os.walk(file_dir):
        print(subdir)
        if not (l == "train" and subdir == file_dir + "/noise"):
            for file in files:
                im = Image.open(os.path.join(subdir, file))
                im = np.expand_dims(im, axis=0)
                print(subdir)
                s = subdir.replace(file_dir, l)
                if not os.path.exists(s):
                    os.mkdir(s)
                imageGen = train_image_generator.flow(im, batch_size=1, save_to_dir=s,save_prefix="im2", save_format="jpg")
                total = 0

                for image in imageGen:
                    total += 1
                    if total == 15:
                        break
shutil.copytree("GRAM_red_100/noise", "train/noise")                        
'''


im_gen = ImageDataGenerator(rescale = 1./255, validation_split = 0.25)
file_dir = "train_v1"

train_data_gen = im_gen.flow_from_directory(seed = 218, batch_size=batch_size,
                                                           directory=file_dir,
                                                           shuffle=True,
                                                           target_size=(img_height, img_width),
                                                           #save_to_dir = "aug",
                                                           class_mode='categorical', subset='training')
val_data_gen = im_gen.flow_from_directory(seed = 218, batch_size=batch_size,
                                                           directory=file_dir,
                                                           shuffle=True,
                                                           target_size=(img_height, img_width),
                                                           #save_to_dir = "aug",
                                                           class_mode='categorical', subset = 'validation')

print(train_data_gen.class_indices)
num_classes=len(train_data_gen.class_indices)
print(train_data_gen.n)
print(train_data_gen.classes.shape[0])
class_weight = {}
for lab in train_data_gen.class_indices.values():
    class_weight[lab] = (1. * train_data_gen.classes.shape[0])/(train_data_gen.classes == lab).sum()
print(class_weight)

sc = 32
model_n = resnet152_model(img_width, img_height, sc, 3, 6)
w_p="weights/weights" + str(sc) + ".h5"
model_n.load_weights(w_p, by_name=True)
model_n.compile(optimizer=keras.optimizers.Adam(), loss='categorical_crossentropy',metrics=['accuracy'])


for par in range(5, 6):
    sc = 2**par
    base_learning_rate = 0.0001
    weights_path="weights" + str(sc) + ".h5"
    checkpoint = ModelCheckpoint(weights_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]
    num_channels=3
    num_classes = len(train_data_gen.class_indices)
    model = resnet152_model(img_width, img_height, sc, num_channels, len(train_data_gen.class_indices))
    if os.path.exists(weights_path):
        model.load_weights(weights_path, by_name=True)
    #print("k = " + str(int(k * len(model.layers) / 10.0)))

    '''
    model = Sequential()

    model.add(Conv2D(24, (3,3),
                     padding='valid',
                     input_shape=(img_height, img_width, 3),
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(BatchNormalization())
    model.add(Conv2D(24, (3,3),
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(16, (3,3),
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(BatchNormalization())
    model.add(Conv2D(12, (3,3),
                     activation='relu'))
    model.add(Flatten())
    model.add(Dense(units=32, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(units=16, activation='relu'))
    model.add(Dense(num_classes, activation='softmax', name='fc8'))
    '''
    
    
    model.compile(optimizer=keras.optimizers.Adam(lr=base_learning_rate), loss='categorical_crossentropy',
                  metrics=['accuracy'])
    #model.summary()
    #print(model.layers[20].get_weights())
    if not os.path.exists(weights_path):
        history = model.fit_generator(
            train_data_gen,
            steps_per_epoch=train_data_gen.n//train_data_gen.batch_size,
            epochs=32,

            verbose=1,
            callbacks = callbacks_list,
            validation_data=val_data_gen,
            validation_steps=val_data_gen.n//val_data_gen.batch_size)
            #, class_weight=class_weight)


        model.load_weights(weights_path, by_name=True)

    
    '''
    test = im_gen.flow_from_directory(directory="test_n", target_size=(img_height, img_width),#save_to_dir = "aug",
                                                           class_mode='categorical', batch_size=1, shuffle=False, seed = 228)
    step_test=test.n//test.batch_size
    s = 0
    l = []
    for i in range(0, step_test):
        #print("i = " + str(i))
        (x, y) = test.next()
        #print(x)
        #print("y = " + str(y))
        p = model.predict(x)
        l.append([np.argmax(y), np.argmax(p)])
        #print("i = " + str(i))
        #print(p)
        z = (p == np.max(p))
        s += (z == y).all()
    print(s)
    print(model.evaluate_generator(test, step_test, workers=1, pickle_safe=True)[1])
    print(s * 1. / test.n)
    df = pd.DataFrame(l, columns=['y_Actual','y_Predicted'])
    print("conf:")
    confusion_matrix = pd.crosstab(df['y_Actual'], df['y_Predicted'], rownames=['Actual'], colnames=['Predicted'])
    print (confusion_matrix)
    '''
    
    
    print("sc = " + str(sc))
    file_dir = "test2/"
    #file_dir = "test_images/"
    s = 0
    t = 0
    for subdir, dirs, files in os.walk(file_dir):
        for file in files: 
            cl = -1
            if file.find("NEG_COCCI") >= 0:
                cl = 0
            elif file.find("NEGATIVE_BACILLI") >= 0:
                cl = 1
            elif file.find("POS_BACILLI") >= 0:
                cl = 2
            elif file.find("COCCI_CHAINS") >= 0:
                cl = 3
            else:
                cl = 4
            print(str(file) + " class: " + str(cl))
            s += 1
            pred_noise(model_n, os.path.join(subdir, file))
            pred = pred_on_image(model, os.path.join(subdir, file))
            t += (pred == cl)

    
    #print("precision: " + str(t * 1. / s))
   
#pred_noise(model_n, "test_images/NEG_COCCI_9.jpg")
#pred_on_image(model, "test_images/NEG_COCCI_9.jpg")

'''
test.reset()
    
#step_test = val_data_gen.n // val_data_gen.batch_size
print(step_test)
print("classes = " + str(test.classes))
print("filenames = " + str(test.filenames))
print("index_array = " + str(test.index_array))
#classes = test.classes[test.index_array]
#classes = np_utils.to_categorical( classes, num_classes)
#classes = np.squeeze(classes.T)

pred = model.predict_generator(test, step_test)
print("pred = " + str(pred))
#print(pred.shape)
y_pred = np.argmax(pred, axis = -1)
print("y_pred = " + str(y_pred))
print(sum(y_pred == test.classes) * 1. / y_pred.shape[0])
print("*****************************************************")
trainable_count = int(np.sum([K.count_params(p) for p in set(model.trainable_weights)]))
'''