
import argparse
import os.path
import torch
import pdb
import numpy as np
parser = argparse.ArgumentParser()

parser.add_argument('--i', default='raw/cars196.pt', help='input directory')
parser.add_argument('--o', default='cars196.pt', help='output file')
parser.add_argument('--n_tasks', default=5, type=int, help='number of tasks')
parser.add_argument('--seed', default=0, type=int, help='random seed')
args = parser.parse_args()

torch.manual_seed(args.seed)

tasks_tr = []
tasks_te = []

x_tr, y_tr, x_te, y_te = torch.load(os.path.join(args.i))
x_tr = x_tr.float().view(x_tr.size(0), -1)
x_te = x_te.float().view(x_te.size(0), -1)

cpt = int(196 / args.n_tasks)

for t in range(args.n_tasks):
    c1 = t * cpt
    c2 = (t + 1) * cpt
    i_tr = ((y_tr >= c1) & (y_tr < c2)).nonzero().view(-1)
    i_te = ((y_te >= c1) & (y_te < c2)).nonzero().view(-1)
    tasks_tr.append([(c1, c2), x_tr[i_tr].clone(), y_tr[i_tr].clone()])
    tasks_te.append([(c1, c2), x_te[i_te].clone(), y_te[i_te].clone()])

torch.save([tasks_tr, tasks_te], args.o)
print("save data")