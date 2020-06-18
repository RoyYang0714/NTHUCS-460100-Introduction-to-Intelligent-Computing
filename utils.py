import cv2, csv
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

from keras.preprocessing.image import img_to_array
from tensorflow import keras

def read_data(csv_path, data_kind, GAN_csv, figure_size):

    # Read given data
    csvfile = open(csv_path)
    reader = csv.reader(csvfile)

    next(reader)

    labels = []

    for line in reader:
        data_path = "./{}/".format(data_kind) + line[0]
        labels.append([data_path, line[1]])

    csvfile.close()

    #  Read generated data
    if GAN_csv:
        csvfile = open(GAN_csv)
        reader = csv.reader(csvfile)

        next(reader)

        for line in reader:
            data_path = "./DCGAN/Type_{}/Fake/".format(line[1]) + line[0]
            labels.append([data_path, line[1]])

        csvfile.close()

    for i in range(len(labels)):
        labels[i][1] = labels[i][1].replace("A", "0")
        labels[i][1] = labels[i][1].replace("B", "1")
        labels[i][1] = labels[i][1].replace("C", "2")

    X = []
    y = []

    random.shuffle(labels)

    print("Reading {} Dataset".format(data_kind))

    for data in tqdm(labels ,position=0, leave=True):
        img = cv2.imread(data[0])
        res = cv2.resize(img,(figure_size, figure_size), interpolation=cv2.INTER_LINEAR)
        res = img_to_array(res)
        X.append(res)    
        y.append(data[1])
    
    labels.clear()
    
    X = np.asarray(X)
    y = np.asarray(y)

    return X, y

def plot_learning_curves(history, model_name, GAN):

    pd.DataFrame(history.history).plot(figsize=(8,5))
    plt.grid(True)
    plt.gca().set_ylim(0,1)

    if GAN:
        plt.savefig('./learning_curve/{}_DCGAN_learning_curve.png'.format(model_name))
    else:
        plt.savefig('./learning_curve/{}_learning_curve.png'.format(model_name))

def test_acc(X_dev, Y_dev, figure_size, checkpoint_path):

    model = keras.models.load_model(checkpoint_path)

    y_pred = model.predict(X_dev)

    count = 0
    for i in range(len(y_pred)):
        if(np.argmax(y_pred[i]) == np.argmax(Y_dev[i])):
            count += 1

    score = count/len(y_pred)

    print('accurancy: %.2f%s' % (score*100, '%'))
            