import os
import pandas as pd
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
print(label_list)
le = LabelEncoder()
le = le.fit(label_list['col'])
label_list['col'] = le.transform(label_list['col'])
print(label_list.drop_duplicates())