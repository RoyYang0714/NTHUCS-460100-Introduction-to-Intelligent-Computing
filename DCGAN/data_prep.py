import csv, os
from tqdm import tqdm
from argparse import ArgumentParser

parser = ArgumentParser()

parser.add_argument("-d", dest="data_type",help="datset type", default="Train")

args = parser.parse_args()

csvfile = open("../{}.csv".format(args.data_type))
reader = csv.reader(csvfile)

next(reader)

labels = []

for line in reader:
    labels.append([line[0], line[1]])

csvfile.close() 

for i in range(len(labels)):
    labels[i][1] = labels[i][1].replace("A", "0")
    labels[i][1] = labels[i][1].replace("B", "1")
    labels[i][1] = labels[i][1].replace("C", "2")

for data in tqdm(labels ,position=0, leave=True):
    if data[1] == "0":
        os.system('cp ../{}/{} ./Type_0/{}/'.format(args.data_type, data[0], args.data_type))
    elif data[1] == "1":
        os.system('cp ../{}/{} ./Type_1/{}/'.format(args.data_type, data[0], args.data_type))
    elif data[1] == "2":
        os.system('cp ../{}/{} ./Type_2/{}/'.format(args.data_type, data[0], args.data_type))
    else:
        pass