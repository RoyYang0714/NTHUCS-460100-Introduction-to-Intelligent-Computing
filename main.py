from argparse import ArgumentParser
import os, csv
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.utils.np_utils import to_categorical

from utils import read_data, plot_learning_curves, test_acc
from Adaboost import ADABoost

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

figure_size = 64

if args.GAN:
    GAN_csv = './GAN.csv'
else:
    GAN_csv = None

# Data Reading
X_train, Y_train = read_data('./Train.csv', 'Train', GAN_csv, figure_size)
X_train = X_train.astype(np.float32)
Y_train = tf.strings.to_number(Y_train, out_type=tf.float32)
Y_train = to_categorical(Y_train, num_classes = 3)

X_dev, Y_dev = read_data('./Dev.csv', 'Dev', None, figure_size)
X_dev = X_dev.astype(np.float32)
Y_dev = tf.strings.to_number(Y_dev, out_type=tf.float32)
Y_dev = to_categorical(Y_dev, num_classes = 3)

X_train = X_train / 255
X_dev /= 255

# Bossting Models
model_list = ['own_model', 'MobileNetV2', 'ResNet50V2', 'VGG16', 'Xception', 'DenseNet169']

ada = ADABoost(X_train, Y_train, X_dev, model_list)

y_pred = ada.adaboost()

count = 0

for i in range(len(y_pred)):
    if(np.argmax(y_pred[i]) == np.argmax(Y_dev[i])):
        count += 1

score = count/len(y_pred)

print('Dev accurancy: %.2f%s' % (score*100, '%'))
