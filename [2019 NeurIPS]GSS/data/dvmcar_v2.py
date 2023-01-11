from tkinter.messagebox import NO
import cv2
import argparse
import os.path
import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
parser = argparse.ArgumentParser()

parser.add_argument('--i', default='raw/dvmcar.pt', help='input directory')
parser.add_argument('--o', default='dvmcar.pt', type=str, help='output file')
parser.add_argument('--p', default='raw/dvm-car/', help='output file')
parser.add_argument('--n_tasks', default=5, type=int, help='number of tasks')
parser.add_argument('--seed', default=0, type=int, help='random seed')
args = parser.parse_args()

torch.manual_seed(args.seed)

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((128, 128)),
    transforms.ToTensor()
    ])

class Dataset(Dataset):
    def __init__(self, image_path, target_idx, transform):
        self.image_list = image_path
        self.target_idx = target_idx
        self.transform = transform

    def __len__(self):
        return len(self.target_idx)

    def __getitem__(self, idx):
        image = cv2.imread(args.p + self.image_list[self.target_idx[idx]])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transform(image)

        return image

def load_datasets(data_loader):
    image_list = None
    
    for idx, data in enumerate(data_loader):
        print(idx)
        if image_list is None:
            image_list = data
        else:
            image_list = torch.cat((image_list, data), 0)

    return image_list

x_tr, y_tr, x_te, y_te = torch.load(os.path.join(args.i))

cpt = int((y_tr.max() + 1) / args.n_tasks)

for t in range(args.n_tasks):
    tasks_tr = []
    tasks_te = []

    c1 = t * cpt
    c2 = (t + 1) * cpt
    i_tr = ((y_tr >= c1) & (y_tr < c2)).nonzero().view(-1)
    i_te = ((y_te >= c1) & (y_te < c2)).nonzero().view(-1)
    
    print("train")
    TrainDataset = Dataset(x_tr, i_tr, transform)
    TrainLoader = DataLoader(TrainDataset , batch_size=4096, shuffle=False, num_workers=4)
    data_tr = load_datasets(TrainLoader)
    print("test")
    TestDataset = Dataset(x_te, i_te, transform)
    TestLoader = DataLoader(TestDataset , batch_size=2048, shuffle=False, num_workers=4)
    data_te = load_datasets(TestLoader)

    data_tr = data_tr.float().view(data_tr.size(0), -1)
    data_te = data_te.float().view(data_te.size(0), -1)

    tasks_tr.append([(c1, c2), data_tr, y_tr[i_tr].clone()])
    tasks_te.append([(c1, c2), data_te, y_te[i_te].clone()])
    torch.save([tasks_tr, tasks_te], "dvmcar" + str(t) + ".pt")

# torch.save([tasks_tr, tasks_te], args.o)
print("save data")