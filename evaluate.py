import argparse

import tensorflow as tf
import yaml

print(tf.__version__)
from tensorflow.keras import layers
from tensorflow.keras import models
import tensorflow_addons as tfa
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import statistics
import os
import json
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import cv2
import shutil
from sklearn.model_selection import train_test_split
from custom_layers.scale_layer import Scale
from easydict import EasyDict as edict

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run network training')
    parser.add_argument('net_type', type=str, default=None, help='res or eff')
    parser.add_argument('data_path', type=str, default=None, help='Path to the dataset')
    args = parser.parse_args()
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")) as f:
        cfg = edict(yaml.safe_load(f))
    batch_size = 1
    net = args.net_type
    file_dir = args.data_path
    sc = 4
    im_gen = ImageDataGenerator(rescale=1. / 255, validation_split=0.3)
    if (net == 'res'):
        model = load_model('resnet' + str(sc) + '.h5', custom_objects={'Scale': Scale})
    else:
        model = load_model('efficientnetb1-baseline.h5')
        model.trainable = False
    # model.summary()

    test = im_gen.flow_from_directory(directory=os.path.join(file_dir, "test"),
                                      target_size=(cfg.img_height, cfg.img_width),
                                      class_mode='categorical', batch_size=1, shuffle=False, seed=228)
    step_test = test.n // test.batch_size
    print("testing batch size = 1")
    s = 0
    l = []
    for i in range(0, step_test):
        (x, y) = test.next()

        p = model.predict(x)
        # print("i = " + str(i))
        z = (np.argmax(p, axis=1) == np.argmax(y, axis=1)).sum()
        lst = [test.filenames[i]]
        for k in range(0, p.shape[1]):
            lst.append(p[0][k])
        lst.append(np.argmax(p))
        # print("lst = " + str(lst))
        l.append(lst)
        s += z
    df = pd.DataFrame(l, columns=["file", 1, 2, 3, "pred"])
    df.to_csv(net + ".csv", index=False)
    print(df)
    print("accuracy = {}, total images = {}".format(s * 1. / test.n, test.n))
    print("evaluating on test generator")
    print(model.evaluate_generator(test, step_test, workers=1))
