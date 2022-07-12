import tensorflow as tf
print(tf.__version__)
from tensorflow.keras import layers
from tensorflow.keras import models
import tensorflow_addons as tfa
import statistics
import os
import json
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt

import cv2
import shutil
from sklearn.model_selection import train_test_split

base_model = tf.keras.applications.EfficientNetB0(include_top=False, weights='imagenet')
base_model.trainabe = True
AUTOTUNE = tf.data.experimental.AUTOTUNE

CONFIG = dict (
    num_labels = 5,
    train_val_split = 0.2,
    img_width = 150,
    img_height = 150,
    batch_size = 1,
    epochs = 10,
    learning_rate = 0.001,
    dropout = 0.2,
    architecture = "CNN",
    infra = "Kaggle",
    competition = 'canada_proj',
    _wandb_kernel = 'ayut'
)
label_to_id = {
    'NEGATIVE COCCI': 0,
    'NEGATIVE BACILLI': 1,
    'POSITIVE BACILLI': 2,
    'COCCI CHAINS': 3,
    'COCCI CLUMPS': 4,
}

path_train = "5 classes_ns/train"
id_to_label = {value:key for key, value in label_to_id.items()}
df = pd.DataFrame(columns = ['path', 'label'])
for subdir, dirs, files in os.walk(path_train):
    print(subdir)
    for file in files:
        l = subdir[subdir.rfind('train')+len('train')+1:]
        df = df.append({'path':os.path.join(subdir, file), 'label':label_to_id[l]}, ignore_index=True)
print(df.head())
df = df.sample(frac=1).reset_index(drop=True)
df.label = df.label.astype('int32')
df.to_csv('train_df.csv', index=False)

df = pd.read_csv('train_df.csv')
print(df.head())
class_weight = {}
for lab in range(0, CONFIG['num_labels']):
    class_weight[lab] = df.label.count()/df[df.label == lab].label.count()
'''
dir = "perfect split"
im_df = pd.DataFrame(columns = ['path', 'label'])
for subdir, dirs, files in os.walk(dir):
    if subdir.find("bacterias") > 0:
        if subdir.find("COCCI_CHAINS") > 0:
            cl = 3
        elif subdir.find("COCCI_CLUMPS") > 0:
            cl = 4
        elif subdir.find("NEGATIVE_BACILLI") > 0:
            cl = 1
        elif subdir.find("POS_BACILLI") > 0:
            cl = 2
        else:
            cl = 0
        for file in files: 
            im_df = im_df.append({'path':os.path.join(subdir, file), 'label':cl}, ignore_index=True)
im_df = im_df.sample(frac=1).reset_index(drop=True)
im_df.label = im_df.label.astype('int32')
im_df.to_csv("perfect split.csv", index=False)
'''
im_df = pd.read_csv("perfect split.csv")

@tf.function
def decode_image(image):
    # Convert the compressed string to a 3D uint8 tensor
    image = tf.image.decode_jpeg(image, channels=3)
    
    # Normalize image
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    
    # Resize the image to the desired size
    return image

@tf.function
def load_image(df_dict):
    # Load image
    image = tf.io.read_file(df_dict['path'])
    image = decode_image(image)
    
    # Resize image
    image = tf.image.resize(image, (CONFIG['img_height'], CONFIG['img_width']))
    
    # Parse label
    #label = tf.strings.split(df_dict['labels'], sep='')
    label = df_dict['label']
    label = tf.one_hot(indices=label, depth=CONFIG['num_labels'])
    
    return image, label

AUTOTUNE = tf.data.AUTOTUNE

trainloader = tf.data.Dataset.from_tensor_slices(dict(df))
validloader = tf.data.Dataset.from_tensor_slices(dict(im_df))

trainloader = (
    trainloader
    .shuffle(1024)
    .map(load_image, num_parallel_calls=AUTOTUNE)
    .batch(CONFIG['batch_size'])
    .prefetch(AUTOTUNE)
)

validloader = (
    validloader
    .map(load_image, num_parallel_calls=AUTOTUNE)
    .batch(CONFIG['batch_size'])
    .prefetch(AUTOTUNE)
)
image_batch, label_batch = next(iter(trainloader))
print(image_batch[0])
def get_model():
    base_model = tf.keras.applications.EfficientNetB0(include_top=False, weights='imagenet')
    base_model.trainable = True

    inputs = layers.Input((CONFIG['img_height'], CONFIG['img_width'], 3))
    x = base_model(inputs, training=True)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(CONFIG['dropout'])(x)
    outputs = layers.Dense(len(label_to_id), activation='sigmoid')(x)
    
    return models.Model(inputs, outputs)

# Model sanity check
tf.keras.backend.clear_session()
model = get_model()
model.summary()


# Compile model
optimizer = tf.keras.optimizers.Adam(learning_rate=CONFIG['learning_rate'])
model.compile(optimizer, 
              loss=tfa.losses.SigmoidFocalCrossEntropy(), 
              metrics=[tf.keras.metrics.AUC(multi_label=True), 'accuracy'])
earlystopper = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=3, verbose=1, mode='min',
    restore_best_weights=True
)

# Train
model.fit(trainloader, 
          epochs=CONFIG['epochs'],
          validation_data=validloader,
          callbacks=[earlystopper])
          #, class_weight=class_weight)

model.save('efficientnetb0-baseline.h5')    
t = 0
for i in range(0, len(im_df), 1):
    prob_df = im_df.iloc[i:(i+1), :]
    cl = prob_df.iloc[:, 1].to_numpy()
    tl = tf.data.Dataset.from_tensor_slices(dict(prob_df))

    tl = (
        tl
        .map(load_image, num_parallel_calls=AUTOTUNE)
        .batch(CONFIG['batch_size'])
        .prefetch(AUTOTUNE)
    )

    p = model.predict(tl)
    '''
    print(p)
    print(cl)
    print(np.argmax(p, axis = 1))
    '''
    t += (np.argmax(p, axis = 1) == cl)

print(t * 1. / len(im_df))

t = 0
for i in range(0, len(im_df)):
    image = tf.io.read_file(im_df.iloc[i, 0])
    #print(im_df.iloc[i, :])
    image = decode_image(image)
    image = tf.image.resize(image, (CONFIG['img_height'], CONFIG['img_width']))
    label = im_df.iloc[i, 1]
    p = model.predict(np.expand_dims(image, axis = 0))
    #print(p)
    t += (np.argmax(p, axis = 1) == label)
print(t * 1. / len(im_df))