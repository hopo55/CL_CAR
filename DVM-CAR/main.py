from cProfile import label
import os
from tkinter import Label
from tkinter.tix import Tree
from pyparsing import traceParseAction
import torch
import numpy as np
import pandas as pd
from data.loader import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

seed = 0
data_path = "DVM-CAR/data/resized_DVM/"
label_path = 'DVM-CAR/data/label/Ad_table.csv'
save_path = "DVM-CAR/data/"

file_list = []
target_list = []
break_point = False

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((128, 128)),
    transforms.ToTensor()
    ])

for (root, directories, files) in os.walk(data_path):
    if break_point:
        break
    for file in files:
        file_root = root.split(os.sep)[2:]
        file_root = os.sep.join(file_root)
        file_path = os.path.join(file_root, file)
        root_split = str(root).split(os.path.sep)[3:-1]
        # if int((root).split(os.path.sep)[5]) < 2015:
        #     break;
        root_split = " ".join(root_split)
        target_list.append(root_split)
        file_list.append(file_path)
        if len(file_list) == 60000:
            break_point = True
            break

# print(file_list)
print(len(file_list))
label_list = pd.DataFrame(target_list, columns=['class'])
# print(label_list)
le = LabelEncoder()
le = le.fit(label_list['class'])
label_list['class'] = le.transform(label_list['class'])
print(label_list.drop_duplicates())

train_image, test_image, train_target, test_target = train_test_split(file_list, label_list, test_size=0.2, random_state=seed, shuffle=True, stratify=label_list)

# torch.save((train_image, train_target, test_image, test_target), save_path + 'dvmcar.pt')
# print('save dvmcar.pt')

TrainDataset = Dataset(save_path, train_image, train_target, transform)
TrainLoader = DataLoader(TrainDataset , batch_size=512, shuffle=True, num_workers=2)

TestDataset = Dataset(save_path, test_image, test_target, transform)
TestLoader = DataLoader(TestDataset , batch_size=512, shuffle=False, num_workers=2)

x_tr = None

for idx, (data, target) in enumerate(TrainLoader):
    # print(idx)
    if x_tr is None:
        x_tr = data
        y_tr = target
    else:
        x_tr = torch.cat((x_tr, data), 0)
        y_tr = torch.cat((y_tr, target), 0)

print("x_tr shape : ", x_tr.shape)
# torch.save((x_tr, y_tr), save_path + 'dvmcar_train.pt')
# print("save dvmcar_train.pt")

# del x_tr
# del y_tr

x_te = None

for idx, (data, target) in enumerate(TestLoader):
    # print(idx)
    if x_te is None:
        x_te = data
        y_te = target
    else:
        x_te = torch.cat((x_te, data), 0)
        y_te = torch.cat((y_te, target), 0)

x_te=x_te[0:2000]
y_te=y_te[0:2000]
print("x_te shape : ", x_te.shape)
print(y_te.min())
print(y_te.max())
# torch.save((x_te, y_te), save_path + 'dvmcar.pt')
# print("save dvmcar_test.pt")
torch.save((x_tr, y_tr, x_te, y_te), save_path + 'dvmcar.pt')
print('save dvmcar.pt')

