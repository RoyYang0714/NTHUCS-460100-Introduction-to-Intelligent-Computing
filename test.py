from argparse import ArgumentParser
import cv2, csv
from tqdm import tqdm
import numpy as np

import tensorflow as tf
from tensorflow import keras
from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import img_to_array

from utils import read_data
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

if args.GAN:
    GAN_csv = './GAN.csv'
else:
    GAN_csv = None

# Setting
figure_size = 64
test_file = './Test.csv'

# Train Data Reading
X_train, Y_train = read_data('./Train.csv', 'Train', GAN_csv, figure_size)
X_train = X_train.astype(np.float32) / 255
Y_train = tf.strings.to_number(Y_train, out_type=tf.float32)
Y_train = to_categorical(Y_train, num_classes = 3)

# Test Data Reading
csvfile = open(test_file)
reader = csv.reader(csvfile)

next(reader)

img_list = []

for line in reader:
    img_list.append(line[0])

csvfile.close()

X = []

print('Reading Test Dataset')

for data in tqdm(img_list ,position=0, leave=True):
    img = cv2.imread('Program Test/' + data + '.jpg')
    res = cv2.resize(img,(figure_size, figure_size), interpolation=cv2.INTER_LINEAR)
    res = img_to_array(res)
    X.append(res)    

X = np.asarray(X)

X_test = X.astype(np.float32) / 255

# Boosting Model
model_list = ['own_model', 'MobileNetV2', 'ResNet50V2', 'VGG16', 'Xception', 'DenseNet169']

ada = ADABoost(X_train, Y_train, X_test, model_list)

y_pred = ada.adaboost()

with open('./Result.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['ImageID','PredictedLabel'])

    print("Writing Result")

    for i in tqdm(range(len(img_list)) ,position=0, leave=True):
        writer.writerow([img_list[i],int(np.argmax(y_pred[i]))+1])
