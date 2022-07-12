# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
#from tensorflow import keras
import keras
#import sklearn
from tensorflow.keras.models import Sequential
import shutil
import numpy as np
import os
import json
import random
import matplotlib.pyplot as plt
#from tensorflow.keras.callbacks import ModelCheckpoint
from keras.callbacks import ModelCheckpoint
from PIL import Image
from matplotlib.pyplot import imread
import cv2
import pandas as pd
#from tensorflow.keras import backend as K
from keras.utils import np_utils
from keras.models import load_model
from custom_layers.scale_layer import Scale

#from PIL import ImageDraw, Image
batch_size = 32
img_height = img_width = 150
sz = 150
    
def pred_noise(model, im_path, dir, sz):
    temp_dir = "temp" + str(model.count_params())
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir, ignore_errors=True)
    if not os.path.exists(temp_dir):
        os.mkdir(temp_dir)
    save_dir = dir + "/" + im_path[im_path.find("/") + 1:im_path.find(".")] 

    if os.path.exists(save_dir):
        shutil.rmtree(save_dir, ignore_errors=True)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    if not os.path.exists(save_dir + "/bacterias"):
        os.mkdir(save_dir + "/bacterias")
    if not os.path.exists(save_dir + "/noise"):
        os.mkdir(save_dir + "/noise")
    img = cv2.imread(im_path)
    v = []
    for i in range(0, img.shape[0]):
        v.append(img[i, :, :].mean())
    q0 = np.quantile(v, 0.05)

    v = []
    for i in range(0, img.shape[1]):
        v.append(img[:, i, :].mean())
    q1 = np.quantile(v, 0.05)

    lw = 0
    while(img[lw, :, :].mean() < 10):
        lw += 1
    h = img.shape[0] - 1
    while(img[h, :, :].mean() < 10):
        h -= 1
    lf = 0
    while(img[:, lf, :].mean() < 10):
        lf += 1
    r = img.shape[1] - 1
    while(img[:, r, :].mean() < 10):
        r -= 1
    img2 = img[lw:h, lf:r, :].copy()
    x_gr = sz
    y_gr = sz
    
    for i in range(0, int((img2.shape[0] - 21)/ y_gr)):
        for j in range(0, int((img2.shape[1] - 21)/ x_gr)):
            pred = []
            for x_off in range(0, 25, 5 if (sz == 100) else 10):
                for y_off in range(0, 25, 5 if (sz == 100) else 10):
            
                    img_p = img2[y_off + i*y_gr:y_off + (i+1)*y_gr, x_off + j*x_gr:x_off + (j+1)*x_gr, :]
                    #print("i = " + str(i) + ", j = " + str(j) + ", x_off = " + str(x_off) + ", y_off = " + str(y_off))

                    pred.append(img_p)
            pred = np.array(pred)
            best_i = 0
            best_v = 0 
            for k in range(0, pred.shape[0]):
                mask_bone = (pred[k] < 105).sum()
                if mask_bone > best_v:
                    best_v = mask_bone
                    best_i = k
            cv2.imwrite(temp_dir + "/" + str(i) + '_' + str(j) + "_x_off_" + str(x_off) + "_y_off_" + str(y_off) + ".jpg", pred[best_i])

    im_gen = ImageDataGenerator(rescale = 1./255)

    c = 0.7
    b = 0
    n = 0
    for subdir, dirs, files in os.walk(temp_dir):
        for file in files:
            im = Image.open(os.path.join(subdir, file))
            #im = im.resize((100, 100))
            im = np.expand_dims(im, axis=0)
            x = im_gen.flow(im, batch_size=1)
            y = x.next()
            p = model.predict(y)
            if np.argmax(p) == 0:# and p[0][np.argmax(p)] >= c:
                b += 1
                shutil.copy2(temp_dir + "/" + file, save_dir + "/bacterias/" + file)
            else:
                n += 1
                shutil.copy2(temp_dir + "/" + file, save_dir + "/noise/" + file)
    print("b = " + str(b) + ", n = " + str(n))
    shutil.rmtree(temp_dir, ignore_errors=True)
def pred_on_image(model, im_path):

    #save_dir = "pred_images/" + im_path[im_path.find("/") + 1:im_path.find(".")] + "/bacterias"
    save_dir = "perfect split/" + im_path[im_path.find("/") + 1:im_path.find(".")] + "/bacterias"
    img_height = img_weight = 150
    
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
    col = ["file"]
    for i in range(0, p.shape[1]):
        col.append(i)
    col.append("pred")
    df = pd.DataFrame(l, columns = col)
    df.to_csv("preds/pred_" + str(model.count_params()) + im_path[im_path.find("/") + 1:im_path.find(".")] + ".csv", index=False)
    #print(df)
   
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

im_gen = ImageDataGenerator(rescale = 1./255, validation_split = 0.25)
file_dir = "5 classes_ns/train"

train_data_gen = im_gen.flow_from_directory(seed = 218, batch_size=batch_size,
                                                           directory=file_dir,
                                                           shuffle=True,
                                                           target_size=(img_height, img_width),
                                                           #save_to_dir = "aug",
                                                           class_mode='categorical', subset='training')
print(train_data_gen.class_indices)
pref0 = "5c_2"
pref2 = "5c"
w_dir = "weights_" + pref0 + "/"
for par in range(0, 8):
    sc = 2**par
 

    print("sc = " + str(sc))
    prev_best = None
    sc2 = 64
    w_p = None
    for subdir, dirs, files in os.walk("weights_all_vs_all_150/"):
        for file in files:
            if file.find("weights_b_vs_n_" +str(sc2)) >= 0 and file.find(".hdf5") >= 0:
                t = float(file[file.rfind("-") + 1:file.find(".hdf5")])
                if prev_best == None or (prev_best > 0 and t < prev_best):
                    w_p = os.path.join(subdir, file)
                    prev_best = t
    #print("w_p = " + str(w_p))
    #w_p="weights_all_vs_all/weights_b_vs_n_64_checkpoint-04-0.0930.hdf5"
    model_n = load_model(w_p, custom_objects={'Scale': Scale})



    prev_best = None
    for subdir, dirs, files in os.walk(w_dir):
        for file in files:
            if file.find("weights_" + pref2 + "_" +str(sc)) >= 0 and file.find(".hdf5") >= 0:
                t = float(file[file.rfind("-") + 1:file.find(".hdf5")])
                if prev_best == None or (prev_best > 0 and t < prev_best):
                    weights_path = os.path.join(subdir, file)
                    prev_best = t
    if prev_best == None:
        weights_path = w_dir + "weights_5c" + str(sc) + ".h5"
    print(prev_best)

    print(weights_path)


    if os.path.exists(weights_path):
        model = load_model(weights_path, custom_objects={'Scale': Scale})
        print('exists')
    #model = load_model(weights_path, custom_objects={'Scale': Scale})

    file_dir = "test_images/"
    s = 0
    t = 0
    dir = "pred_images"
    dir = "perfect split"
    for subdir, dirs, files in os.walk(file_dir):
        for img in files: 
            for k in train_data_gen.class_indices:
                if k.find(img[:img.find('_')]) >= 0 and k.find(img[img.find('_') + 1 : img.find('_', img.find('_') + 1)]) >= 0:
                    cl = train_data_gen.class_indices[k]
            '''
            if cl == 3:
                continue
            '''
            print(str(img) + " class: " + str(cl))
            s += 1
            im_path = os.path.join(subdir, img)
            if not os.path.exists(dir + "/" + im_path[im_path.find("/") + 1:im_path.find(".")]):

                pred_noise(model_n, im_path, dir, sz)

            pred = pred_on_image(model, os.path.join(subdir, img))
            t += (pred == cl)
            #t += (pred == (cl <= 1))

    print("precision: " + str(t * 1. / s))
    
#pred_noise(model_n, "test_images/NEG_COCCI_9.jpg")
#pred_on_image(model, "test_images/NEG_COCCI_9.jpg")
'''
type = "COCCI CLUMPS"
if type == "NEGATIVE COCCI":
    pref1 = "neg_c"
    pref2 = "nc"  
elif type == "NEGATIVE BACILLI":
    pref1 = "neg_b"
    pref2 = "nb" 
elif type == "POSITIVE BACILLI":
    pref1 = "pos_b"
    pref2 = "pb"
elif type == "COCCI CHAINS":
    pref1 = "coc_ch"
    pref2 = "cch"
else:
    pref1 = "coc_cl"
    pref2 = "ccl"
dir = "preds_" + type
if os.path.exists(dir):
    shutil.rmtree(dir, ignore_errors=True)
if not os.path.exists(dir):
    os.mkdir(dir)
#w_p = "weights_bac_noise/weights_" + pref1 + "/weights_nc64.h5"
#w_p = "weights_bac_noise/weights_" + pref1 + "/weigths_nb32.h5"
#w_p = "weights_bac_noise/weights_" + pref1 + "/weights_pb4.h5"
#w_p = "weights_bac_noise/weights_" + pref1 + "/weights_cch_64_checkpoint-04-0.1942.hdf5"
w_p = "weights_bac_noise/weights_" + pref1 + "/weights_ccl_128_checkpoint-06-0.0426.hdf5"
if w_p.find(".hdf5") >= 0:
    model = load_model(w_p, custom_objects={'Scale': Scale})
else:
    sc = int(w_p[w_p.find(pref2) + len(pref2):w_p.find(".h5")])
    num_channels = 3
    print(sc)
    model = resnet152_model(img_width, img_height, sc, num_channels, 2)
    model.load_weights(w_p)
    base_learning_rate = 0.0001
    model.compile(optimizer=keras.optimizers.Adam(lr=base_learning_rate), loss='categorical_crossentropy',metrics=['accuracy'])
file_dir = "test_images/"
s = 0
t = 0
for subdir, dirs, files in os.walk(file_dir):
    for img in files: 
        print(img)
        pred_noise(model, os.path.join(subdir, img), dir)
'''