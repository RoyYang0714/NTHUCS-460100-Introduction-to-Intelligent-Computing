import numpy as np
from argparse import ArgumentParser

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from tensorflow import keras

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras import utils as np_utils
from keras import optimizers

from utils import read_data, plot_learning_curves, test_acc

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = ArgumentParser()

parser.add_argument("-GAN", dest="GAN", type=str2bool, help="using DCGAN data or not", default=True)

args = parser.parse_args()

# HyperParameters
figure_size = 64
learning_rate = 0.003
batch_size =  64
epochs = 20
drop = 0.2

# Train Data Reading
if args.GAN:
    GAN_csv = './GAN.csv'
else:
    GAN_csv = None

X_train, Y_train = read_data('./Train.csv', 'Train', GAN_csv, figure_size)
X_train = X_train.astype(np.float32) / 255
Y_train = tf.strings.to_number(Y_train, out_type=tf.float32)
Y_train = np_utils.to_categorical(Y_train, num_classes = 3)

X_dev, Y_dev = read_data('./Dev.csv', 'Dev', None, figure_size)
X_dev = X_dev.astype(np.float32) / 255
Y_dev = tf.strings.to_number(Y_dev, out_type=tf.float32)
Y_dev = np_utils.to_categorical(Y_dev, num_classes = 3)

# Model
model = tf.keras.Sequential()

model.add(layers.Conv2D(16, (3,3),
                strides=(1,1),
                input_shape=(figure_size, figure_size, 3),
                padding='same',
                activation='relu',
                kernel_regularizer=tf.keras.regularizers.l2(0.01)
                ))

model.add(layers.BatchNormalization())

model.add(layers.MaxPooling2D(pool_size=(2,2), strides=None))

model.add(layers.Conv2D(32, (3,3),
                strides=(1,1),
                padding='same',
                activation='relu',
                kernel_regularizer=tf.keras.regularizers.l2(0.01)
                ))

model.add(layers.MaxPooling2D(pool_size=(2,2), strides=None))

model.add(layers.Conv2D(64,(3,3),
                strides=(1,1),
                padding='same',
                activation='relu',
                kernel_regularizer=tf.keras.regularizers.l2(0.01)
                ))

model.add(layers.MaxPooling2D(pool_size=(2,2), strides=None))

model.add(layers.Flatten())

model.add(layers.Dense(64, activation='relu'))

model.add(layers.Dense(128, activation='relu'))

model.add(layers.Dropout(0.4))

model.add(layers.Dense(3, activation='softmax'))

model.summary()

adam = optimizers.adam(lr = learning_rate)
model.compile(optimizer='adam',
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=['acc'])

datagen = ImageDataGenerator(
    zca_whitening=False,
    featurewise_std_normalization=True,
    rotation_range=45,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest')

datagen.fit(X_train)

callbacks = []
callbacks.append(tf.keras.callbacks.ModelCheckpoint('./checkpoints/own_model.h5', save_best_only=True, save_weights_only=False))

history = model.fit(
    x = X_train , y = Y_train,
    batch_size = batch_size,
    epochs = epochs,
    validation_split = 0.2,
    callbacks = callbacks,
)

plot_learning_curves(history, 'own_model', args.GAN)

test_acc(X_dev, Y_dev, figure_size, './checkpoints/own_model.h5')
