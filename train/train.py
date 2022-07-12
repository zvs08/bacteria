import tensorflow as tf

print(tf.__version__)
from tensorflow.keras import layers
from tensorflow.keras import models
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
from bac_resnet import resnet152_model, SaveModelCheckpoint
from custom_layers.scale_layer import Scale
import argparse
from pathlib import Path
import yaml
from easydict import EasyDict as edict

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run network training')
    parser.add_argument('net_type', type=str, default=None, help='res or eff')
    parser.add_argument('data_path', type=str, default=None, help='Path to the dataset')
    args = parser.parse_args()

    with open(os.path.join(str(Path(os.path.dirname(os.path.abspath(__file__))).parent), "config.yaml")) as f:
        cfg = edict(yaml.safe_load(f))
    print(cfg)
    file_dir = args.data_path
    print(os.listdir(file_dir))
    net = args.net_type
    im_gen = ImageDataGenerator(rescale=1. / 255, validation_split=0.3)
    train_generator = im_gen.flow_from_directory(seed=218, batch_size=cfg.batch_size,
                                                 directory=os.path.join(file_dir, "train"),
                                                 shuffle=True,
                                                 target_size=(cfg.img_height, cfg.img_width),
                                                 # save_to_dir = "aug",
                                                 class_mode='categorical', subset='training')
    validation_generator = im_gen.flow_from_directory(seed=218, batch_size=cfg.batch_size,
                                                      directory=os.path.join(file_dir, "train"),
                                                      shuffle=True,
                                                      target_size=(cfg.img_height, cfg.img_width),
                                                      # save_to_dir = "aug",
                                                      class_mode='categorical', subset='validation')
    num_classes = len(train_generator.class_indices)
    print(num_classes)

    sc = cfg.scale
    ex = 1
    chkp_path = os.path.join(str(Path(os.path.dirname(os.path.abspath(__file__))).parent), "chekpoints")
    if not os.path.exists(chkp_path):
        os.mkdir(chkp_path)
        os.mkdir(os.path.join(chkp_path, "resnet"))
        os.mkdir(os.path.join(chkp_path, "efficientnet"))
    if (net == 'res'):
        if not os.path.exists(os.path.join(chkp_path, "resnet", 'resnet' + str(sc) + '.h5')):
            ex = 0
            model = resnet152_model(cfg.img_height, cfg.img_width, sc=sc, num_classes=num_classes)
            model.summary()
            chkp_path = os.path.join(chkp_path, "resnet", 'resnet' + str(sc) + '.h5')
            checkpoint = SaveModelCheckpoint(chkp_path, monitor='val_loss', verbose=1,
                                             save_best_only=True, mode='min', prev_best=None)
    if (net == 'eff'):
        if not os.path.exists(os.path.join(chkp_path, "efficientnet", 'efficientnetb1-baseline.h5')):
            ex = 0
            base_model = tf.keras.applications.EfficientNetB1(include_top=False, weights='imagenet')
            base_model.trainable = True
            inputs = layers.Input((cfg.img_height, cfg.img_width, 3))
            x = base_model(inputs, training=True)
            x = layers.GlobalAveragePooling2D()(x)
            x = layers.Dropout(0.5)(x)
            outputs = layers.Dense(num_classes, activation='softmax')(x)
            chkp_path = os.path.join(chkp_path, "efficientnet", 'efficientnetb1-baseline.h5')
            ## Compile and run
            model = models.Model(inputs, outputs)
            checkpoint = SaveModelCheckpoint(chkp_path, monitor='val_loss', verbose=1,
                                             save_best_only=True, mode='min', prev_best=None)
            model.summary()
    if (ex == 0):
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        model.compile(optimizer,
                      loss='categorical_crossentropy',  # tfa.losses.SigmoidFocalCrossEntropy(),
                      metrics=['accuracy'])
        earlystopper = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=3, verbose=0, mode='min',
            restore_best_weights=True
        )

        callbacks_list = [checkpoint]
        # Train
        model.fit(train_generator,
                  epochs=8,
                  validation_data=validation_generator,
                  callbacks=[callbacks_list])
        # earlystopper])

        # model.save('efficientnetb1-baseline.h5')
