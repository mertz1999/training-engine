from args import train_args
import tensorflow as tf
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers, models, optimizers, callbacks
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
# import keras
from tensorflow import keras



import os

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

# data augmentation and normalization
data_augmentation = tf.keras.Sequential([
    keras.layers.Rescaling(1./255),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])


# build model
def build_model(input_shape, num_classes):
    # Load ConvNeXt base model without the top layer
    base_model = tf.keras.applications.ConvNeXtSmall(include_top=False, input_shape=input_shape, weights=None)
    base_model.trainable = True  # Set to False if you want to freeze layers

    model = models.Sequential([
        data_augmentation,
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=args.opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

model = build_model((32,32,3), 2)



# Callbacks for saving checkpoints and best model
checkpoint_cb = callbacks.ModelCheckpoint(
    os.path.join(args.project,'last.weights.h5'), save_best_only=False, save_weights_only=True,
    monitor='val_loss', verbose=1, save_freq='epoch')

checkpoint_cb_tf = callbacks.ModelCheckpoint(
    os.path.join(args.project,'last.keras'), save_best_only=False, save_weights_only=False,
    monitor='val_loss', verbose=1, save_freq='epoch')

best_model_cb = callbacks.ModelCheckpoint(
    os.path.join(args.project,'best_model.weights.h5'), save_best_only=True, save_weights_only=True,
    monitor='val_loss', verbose=1)


# Train the model
history = model.fit(train_ds, epochs=args.epoch, validation_data=test_ds,
                    callbacks=None)

# save model
model.save_weights(os.path.join(args.project,'last.weights.h5'))




