from asyncio import FastChildWatcher
import torch
import glob
from pickletools import pystring
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from data.loader import read_mat_to_df, Dataset, Dataset_all
from sklearn.model_selection import train_test_split

seed = 0
data_path = 'StanfordCar/data/'

file_list = glob.glob(data_path + 'car_ims/*')
target_list = read_mat_to_df(data_path + 'cars_annos.mat')
print

train_image, test_image, train_target, test_target = train_test_split(file_list, target_list, test_size=0.2, random_state=seed, shuffle=True)

'''
train_foler = glob.glob(data_path + 'cars_train/*')
test_foler = glob.glob(data_path + 'cars_test/*')

train_label = read_mat_to_df(data_path + 'devkit/cars_train_annos.mat')
train_label = train_label.sort_values(['class', 'relative_im_path'], axis=0)

test_label = read_mat_to_df(data_path + 'devkit/cars_test_annos_withlabels.mat')
test_label = test_label.sort_values(['class', 'relative_im_path'], axis=0)
'''

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((128, 128)),
    transforms.ToTensor()
    ])

# TrainDataset = Dataset(data_path, train_label, transform)
TrainDataset = Dataset_all(train_image, train_target, transform)
TrainLoader = DataLoader(TrainDataset , batch_size=512, shuffle=True, num_workers=2)

# TestDataset = Dataset(data_path, test_label, transform)
TestDataset = Dataset_all(test_image, test_target, transform)
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

torch.save((x_tr, y_tr, x_te, y_te), data_path + 'cars196_128.pt')
print('save cars196.pt')