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
from train.bac_resnet import resnet152_model, SaveModelCheckpoint
from PIL import Image
from matplotlib.pyplot import imread
import cv2
import pandas as pd
#from tensorflow.keras import backend as K
from keras import backend as K
from keras.utils import np_utils
from keras.models import load_model
from custom_layers.scale_layer import Scale

#from PIL import ImageDraw, Image
batch_size = 20
img_height = img_width = 150
num_channels = 3

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

type = "5 classes_ns"
print(type)
if type == "NEGATIVE COCCI":
    pref0 = "bac_noise"
    pref1 = "neg_c"
    pref2 = "nc"  
elif type == "NEGATIVE BACILLI":
    pref0 = "bac_noise"
    pref1 = "neg_b"
    pref2 = "nb" 
elif type == "POSITIVE BACILLI":
    pref0 = "bac_noise"
    pref1 = "pos_b"
    pref2 = "pb"
elif type == "COCCI CHAINS":
    pref0 = "bac_noise"
    pref1 = "coc_ch"
    pref2 = "cch"
elif type == "COCCI CLUMPS":
    pref0 = "bac_noise"
    pref1 = "coc_cl"
    pref2 = "ccl"
elif type == "b_vs_n_150":
    pref0 = "all_vs_all_150"
    pref1 = ""
    pref2 = "b_vs_n"
elif type == "5 classes_ns":
    pref0 = "5c"
    pref1 = ""
    pref2 = "5c"
else:
    pref0 = "bac_coc"
    pref1 = ""
    pref2 = "bc"
im_gen = ImageDataGenerator(rescale = 1./255, validation_split = 0.25)
file_dir = type + "/train"


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
    if type.find("5 classes") >= 0:
        class_weight[lab] = (1. * train_data_gen.classes.shape[0])/(train_data_gen.classes == lab).sum()
    else:
        class_weight[lab] = 1
print(class_weight)

ar = [2, 4, 8, 16, 32, 64, 128, 192, 224]

for par in ar:
    sc = 2**par
    sc = par
    print("sc = " + str(sc))
    
    prev_best = None
    num_channels=3
    base_learning_rate = 0.0001
    num_classes = len(train_data_gen.class_indices)
    model = resnet152_model(img_width, img_height, sc, num_channels, len(train_data_gen.class_indices))
    weights_path = ""
    ex_fl = 0
    if pref1 != "":
        w_dir = "weights_" + pref0 + "/weights_" + pref1 + "/"
    else:
        w_dir = "weights_" + pref0 + "_2/"
    
    for subdir, dirs, files in os.walk(w_dir):
        for file in files:
            if file.find("weights_" + pref2 + "_" +str(sc)) >= 0 and file.find(".hdf5") >= 0:
                t = float(file[file.rfind("-") + 1:file.find(".hdf5")])
                if prev_best == None or (prev_best > 0 and t < prev_best):
                    weights_path = os.path.join(subdir, file)
                    prev_best = t
    if prev_best == None:
        weights_path = w_dir + "weights_" + pref2 + str(sc) + ".h5"
    print(weights_path)
    print(prev_best)
    
    if os.path.exists(weights_path):
        model = load_model(weights_path, custom_objects={'Scale': Scale})
        print('exists')
        ex_fl = 1
    '''
    for subdir, dirs, files in os.walk("weights_" + pref0 + "/weights_" + pref1 + "/"):
        for file in files:
            weights_path = os.path.join(subdir, file)
            print(file)
            s = (file.find("_01.") == -1) * file.find(".") + (file.find("_01.") >= 0) * file.find("_01.")
            sc = int(file[file.find("pb")+2:s])
   '''
    weights_path=w_dir + "weights_" + pref2 + "_" + str(sc) + "_checkpoint-{epoch:02d}-{val_loss:.4f}.hdf5"
    checkpoint = SaveModelCheckpoint(weights_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min', prev_best = prev_best)
    callbacks_list = [checkpoint]
    
    
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
    if not ex_fl:
        history = model.fit_generator(
            train_data_gen,
            steps_per_epoch=train_data_gen.n//train_data_gen.batch_size,
            epochs=32,
            verbose=1,
            callbacks = callbacks_list,
            validation_data=val_data_gen,
            validation_steps=val_data_gen.n//val_data_gen.batch_size
            , class_weight=class_weight)
        
        prev_best = None
        for subdir, dirs, files in os.walk(w_dir):
            for file in files:
                if file.find("weights_" + pref2 + "_" +str(sc)) >= 0 and file.find(".hdf5") >= 0:
                    t = float(file[file.rfind("-") + 1:file.find(".hdf5")])
                    if prev_best == None or (prev_best > 0 and t < prev_best):
                        weights_path = os.path.join(subdir, file)
                        prev_best = t
        model = load_model(weights_path, custom_objects={'Scale': Scale})


    #print("sc = " + str(sc))
    test = im_gen.flow_from_directory(directory=type + "/test", target_size=(img_height, img_width),#save_to_dir = "aug",
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
pref0 = "5c"
pref2 = "5c"
par = 6
sc = 2**par
prev_best = None
for subdir, dirs, files in os.walk("weights_" + pref2):
    for file in files:
        if file.find("weights_" + pref2 + "_" +str(sc)) >= 0 and file.find(".hdf5") >= 0:
            t = float(file[file.rfind("-") + 1:file.find(".hdf5")])
            if prev_best == None or (prev_best > 0 and t < prev_best):
                weights_path = os.path.join(subdir, file)
                prev_best = t
model = load_model(weights_path, custom_objects={'Scale': Scale})

print("--------------validation--------------------")
wrong_p = "wrong_val_" + pref2
if not os.path.exists(wrong_p):
    os.mkdir(wrong_p)

test = im_gen.flow_from_directory(directory=type + "/train", target_size=(img_height, img_width),class_mode='categorical',
                                                            subset = 'validation', batch_size=1, shuffle=False, seed = 228)
step_test=test.n//test.batch_size
print("size = " + str(step_test))
s = 0
l = []
print("index_array = " + str(test.index_array))
for i in range(0, step_test):
    
    (x, y) = test.next()
    p = model.predict(x)
    #print("p = " + str(p))
    #print("y = " + str(y))
    l.append([np.argmax(y), np.argmax(p)])
    z = (p == np.max(p))
    s += (z == y).all()
    if np.argmax(y) != np.argmax(p):
        for k in train_data_gen.class_indices:
            if train_data_gen.class_indices[k] == np.argmax(p):
                p_c = k
            if train_data_gen.class_indices[k] == np.argmax(y):
                a_c = k  
        print(test.filenames[i])
        name = test.filenames[i][test.filenames[i].find("/")+1:]
        dir = wrong_p + "/" + a_c + " predicted as " + p_c
        if not os.path.exists(dir):
            os.mkdir(dir)
        shutil.copy2(type + "/train/" + test.filenames[i], dir + "/" + name)
print(s)
print(model.evaluate_generator(test, step_test, workers=1, pickle_safe=True)[1])
print(s * 1. / test.n)
df = pd.DataFrame(l, columns=['y_Actual','y_Predicted'])
print("conf:")
confusion_matrix = pd.crosstab(df['y_Actual'], df['y_Predicted'], rownames=['Actual'], colnames=['Predicted'])
print (confusion_matrix) 
'''