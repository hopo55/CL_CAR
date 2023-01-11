import os
from tkinter import Label
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
data_path = "CompCars/data/image"
save_path = "CompCars/data/"

file_list = []
target_list = []

for (root, directories, files) in os.walk(data_path):
    for file in files:
        file_path = os.path.join(root, file)
        root_split = str(root).split(os.path.sep)[3:]
        if root_split[2] == "unknown" or int(root_split[2]) > 2022:
            break
        root_split = " ".join(root_split)
        target_list.append(root_split)
        file_list.append(file_path)

label_list = pd.DataFrame(target_list, columns=['col'])
le = LabelEncoder()
le = le.fit(label_list['col'])
label_list['col'] = le.transform(label_list['col'])
# print(label_list.drop_duplicates())
# print(label_list.drop_duplicates().min())
# print(label_list.drop_duplicates().max())

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((128, 128)),
    transforms.ToTensor()
    ])

train_image, test_image, train_target, test_target = train_test_split(file_list, label_list, test_size=0.2, random_state=seed, shuffle=True)

# torch.save((x_tr, y_tr, x_te, y_te), save_path + 'compcars.pt')

TrainDataset = Dataset(train_image, train_target, transform)
TrainLoader = DataLoader(TrainDataset , batch_size=512, shuffle=True, num_workers=2)

TestDataset = Dataset(test_image, test_target, transform)
TestLoader = DataLoader(TestDataset , batch_size=512, shuffle=False, num_workers=2)

x_tr = None

for idx, (data, target) in enumerate(TrainLoader):
    print(idx)
    if x_tr is None:
        x_tr = data
        y_tr = target
    else:
        x_tr = torch.cat((x_tr, data), 0)
        y_tr = torch.cat((y_tr, target), 0)

print(x_tr.shape)
# torch.save((x_tr, y_tr), save_path + 'compcars_train.pt')
# print("save compcars_train.pt")

# del x_tr
# del y_tr

x_te = None

for idx, (data, target) in enumerate(TestLoader):
    print(idx)
    if x_te is None:
        x_te = data
        y_te = target
    else:
        x_te = torch.cat((x_te, data), 0)
        y_te = torch.cat((y_te, target), 0)

print(x_te.shape)
# torch.save((x_te, y_te), save_path + 'compcars_test.pt')
# print("save compcars_test.pt")
torch.save((x_tr, y_tr, x_te, y_te), save_path + 'compcars.pt')
print('save compcars.pt')
