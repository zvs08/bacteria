import argparse

import tensorflow as tf
import yaml

print(tf.__version__)
from tensorflow.keras import layers
from tensorflow.keras import models
import tensorflow_addons as tfa
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import cv2
import shutil
from easydict import EasyDict as edict
from bac_resnet import resnet152_model, SaveModelCheckpoint
from custom_layers.scale_layer import Scale

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run network training')
    parser.add_argument('net_type', type=str, default=None, help='res or eff')
    parser.add_argument('data_path', type=str, default=None, help='Path to the dataset')
    args = parser.parse_args()
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")) as f:
        cfg = edict(yaml.safe_load(f))
    batch_size = 1
    net = args.net_type
    test_dir = args.data_path
    sc = cfg.scale
    im_gen = ImageDataGenerator(rescale=1. / 255, validation_split=0.3)
    if (net == 'res'):
        model = load_model(os.path.join("checkpoints", "resnet", 'resnet' + str(sc) + '.h5'),
                           custom_objects={'Scale': Scale})
    else:
        model = load_model(os.path.join("checkpoints", "efficientnet", 'efficientnetb1-baseline.h5'))
        model.trainable = False

    for subdir, dirs, files in os.walk(test_dir):
        t = 0
        s = 0
        for file in files:
            print(file)
            cl = -1
            if file.find("NB") >= 0:
                cl = 2
            elif file.find("CCH") >= 0:
                cl = 0
            elif file.find("CCL") >= 0:
                cl = 1
            else:
                cl = -1
            print(file + " class: " + str(
                next((name for name, v in train_generator.class_indices.items() if v == cl), None)))
            temp_dir = "temp_" + str(file)
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)
            if not os.path.exists(temp_dir):
                os.mkdir(temp_dir)
            f = []
            img = cv2.imread(os.path.join(subdir, file))
            img_p = img[:int(img.shape[0] / 2), :int(img.shape[1] / 2), :]
            cv2.imwrite(os.path.join(temp_dir, file[:file.find('.')] + "_I.jpg"), img_p)
            f.append(file[:file.find('.')] + "_I.jpg")
            img_p = img[:int(img.shape[0] / 2), int(img.shape[1] / 2):, :]
            cv2.imwrite(os.path.join(temp_dir, file[:file.find('.')] + "_II.jpg"), img_p)
            f.append(file[:file.find('.')] + "_II.jpg")
            img_p = img[int(img.shape[0] / 2):, 0:int(img.shape[1] / 2), :]
            cv2.imwrite(os.path.join(temp_dir, file[:file.find('.')] + "_III.jpg"), img_p)
            f.append(file[:file.find('.')] + "_III.jpg")
            img_p = img[int(img.shape[0] / 2):, int(img.shape[1] / 2):, :]
            cv2.imwrite(os.path.join(temp_dir + "/" + file[:file.find('.')] + "_IV.jpg"), img_p)
            f.append(file[:file.find('.')] + "_IV.jpg")
            f = pd.DataFrame(f, columns=['filename'])
            im_gen = ImageDataGenerator(rescale=1. / 255)
            test = im_gen.flow_from_dataframe(f, directory=temp_dir,
                                              target_size=(CONFIG['img_height'], CONFIG['img_width']),
                                              class_mode=None, batch_size=1, shuffle=False, seed=228)
            step_test = test.n // test.batch_size

            l = []
            z = np.zeros((1, num_classes))
            for i in range(0, step_test):
                x = test.next()
                p = model.predict(x)
                # print("i = " + str(i))
                z += p
                lst = [test.filenames[i]]
                for k in range(0, p.shape[1]):
                    lst.append(p[0][k])
                lst.append(np.argmax(p))
                # print("lst = " + str(lst))
                l.append(lst)
            df = pd.DataFrame(l, columns=["file", 0, 1, 2, "pred"])
            print(z)
            pred = np.argmax(z)
            print(pred)
            print(df)
            print('\n')
            t += 1
            s += (pred == cl)
            shutil.rmtree(temp_dir, ignore_errors=True)
    print(s * 1. / t)
