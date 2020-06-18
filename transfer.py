import numpy as np
from argparse import ArgumentParser

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Conv2D
from tensorflow.keras.optimizers import SGD

from keras.preprocessing.image import ImageDataGenerator
from keras import utils as np_utils
from keras import optimizers
from keras.callbacks import ModelCheckpoint

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

model_list = ['MobileNetV2', 'ResNet50V2', 'VGG16', 'Xception', 'DenseNet169']

# HyperParameters
figure_size = 64
learning_rate = 1e-4
batch_size = 64
epochs = 25

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
for model_name in model_list:

    print("Transfer {} ...".format(model_name))

    pretrained_model = getattr(tf.keras.applications, model_name)
    base_model = pretrained_model(weights='imagenet', include_top=False)

    x = base_model.output
    x = GlobalAveragePooling2D()(x)

    x = Dropout(0.5)(x)
    x = Dense(1024, activation='relu')(x)
    x = Dense(128, activation='relu')(x)
    predictions = Dense(3, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    for layer in base_model.layers:
        layer.trainable = False

    adam = optimizers.adam()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

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
    callbacks.append(ModelCheckpoint('./checkpoints/{}.h5'.format(model_name), save_best_only=True, save_weights_only=False))

    history = model.fit(
        x = X_train , y = Y_train,
        batch_size = batch_size,
        epochs = epochs,
        validation_split = 0.2,
        callbacks = callbacks,
    )

    plot_learning_curves(history, model_name + '_transfer1', args.GAN)

    for layer in model.layers[:-5]:
        layer.trainable = False
    for layer in model.layers[-5:]:
        layer.trainable = True

    model.compile( optimizer = SGD(lr = 0.0001, momentum = 0.9), 
                    loss='categorical_crossentropy', metrics=['acc'])

    callbacks = []
    callbacks.append(ModelCheckpoint('./checkpoints/{}.h5'.format(model_name), save_best_only=True, save_weights_only=False))

    history = model.fit(
        x = X_train , y = Y_train,
        batch_size = batch_size,
        epochs = epochs,
        validation_split = 0.2,
        callbacks = callbacks,
    )

    plot_learning_curves(history, model_name + '_transfer2', args.GAN)

    test_acc(X_dev, Y_dev, figure_size, './checkpoints/{}.h5'.format(model_name))
