from args import train_args
import tensorflow as tf
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras
from inc.inc import make_model

def clear_screen():
    # Check if the operating system is Windows
    if os.name == 'nt':
        os.system('cls')
    else:
        os.system('clear')


# get all settings
args = train_args()


# load dataset (train and val)
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    os.path.join(args.data, 'train'),  # Adjust this path
    seed=123,
    image_size=(args.size, args.size),
    shuffle=True,
    label_mode='categorical',
    batch_size=args.batch)

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    os.path.join(args.data, 'val'),  # Adjust this path
    seed=123,
    image_size=(args.size, args.size),
    shuffle=True,
    label_mode='categorical',
    batch_size=args.batch)

clear_screen()
print('class names : ',train_ds.class_names)

# pre-fetch dataset
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

# make and build model
model = make_model((32,32,3), 2, args.config)
model.compile(optimizer=args.opt, loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(train_ds, epochs=args.epoch, validation_data=test_ds,callbacks=None)

# save model
model.save_weights(os.path.join(args.project,'last.weights.h5'))




