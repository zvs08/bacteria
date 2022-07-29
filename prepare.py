# -*- coding: utf-8 -*-
import argparse

import shutil

import cv2
import numpy as np
import os

'''
script for creating the dataset. Takes one argument - path to the source images. It should have the following structure:
data_root
    COCCI CHAINS
        1.jpg
        2.jpg
        ...
    COCCI CLUMPS
        1.jpg
        2.jpg
        ...
    NEGATIVE BACILLI
        1.jpg
        2.jpg
        ...
The output will produce 3 new folders - train, test, test_images
'''

def rename(path):
    classes = os.listdir(path)
    for cl in classes:
        pref = cl.split(' ')[0][0] + (cl.split(' ')[1][0] if cl.split(' ')[0][0] == 'N' else cl.split(' ')[1][0:2])
        for file in os.listdir(os.path.join(path, cl)):
            os.rename(os.path.join(path, cl, file), os.path.join(path, cl, pref + '_' + file))

def cut_images(path):
    classes = os.listdir(path)
    classes = [c for c in classes if c not in ("train", "test", "test_images")]
    print(classes)
    if not os.path.exists(os.path.join(path, "train")):
        os.mkdir(os.path.join(path, "train"))
    if not os.path.exists(os.path.join(path, "test")):
        os.mkdir(os.path.join(path, "test"))
    if not os.path.exists(os.path.join(path, "test_images")):
        os.mkdir(os.path.join(path, "test_images"))
    for cl in classes:
        files = os.listdir(os.path.join(path, cl))
        train_n = min(len(files) - 2, np.int(0.96 * len(files)))
        for sub in ("train", "test", "test_images"):
            if not os.path.exists(os.path.join(path, sub, cl)):
                os.mkdir(os.path.join(path, sub, cl))
        for j in range(train_n, len(files)):
            shutil.copyfile(os.path.join(path, cl, files[j]), os.path.join(path, "test_images", cl, files[j]))
        for sub in ("train", "test"):
            for j in range(0 if sub == "train" else train_n, train_n if sub == "train" else len(files)):
                img = cv2.imread(os.path.join(path, cl, files[j]))
                img_p = img[:int(img.shape[0] / 2), :int(img.shape[1] / 2), :]
                cv2.imwrite(os.path.join(path, sub, cl, files[j][:files[j].find('.')] + "_I.jpg"), img_p)

                img_p = img[:int(img.shape[0] / 2), int(img.shape[1] / 2):, :]
                cv2.imwrite(os.path.join(path, sub, cl, files[j][:files[j].find('.')] + "_II.jpg"), img_p)
                img_p = img[int(img.shape[0] / 2):, 0:int(img.shape[1] / 2), :]
                cv2.imwrite(os.path.join(path, sub, cl, files[j][:files[j].find('.')] + "_III.jpg"), img_p)
                img_p = img[int(img.shape[0] / 2):, int(img.shape[1] / 2):, :]
                cv2.imwrite(os.path.join(path, sub, cl, files[j][:files[j].find('.')] + "_IV.jpg"), img_p)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path', type=str, default=None, help='Path to the dataset')
    args = parser.parse_args()
    #rename(args.data_path)
    cut_images(args.data_path)