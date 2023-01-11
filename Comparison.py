from tkinter.font import names
import numpy as np
import pandas as pd
from scipy.io import loadmat
from torch import equal

''' Stanford Car'''
# Make, Model, Year
file_path = 'StanfordCar/data/devkit/cars_meta.mat'

cars_meta = loadmat(file_path)
class_names = cars_meta['class_names']  # shape=(1, 196)
class_names = np.transpose(class_names)

cars = []
years = []
bm = []
# year_min = np.min(class_names)

for idx in range(len(class_names)):
    car_name = str(class_names[idx][0][0]).split(' ')[0:-2]
    year = str(class_names[idx][0][0]).split(' ')[-1:]
    bm.append(' '.join(car_name))
    car_name = car_name + year
    years.extend(year)
    cars.append(' '.join(car_name))

years = list(map(int, years))

print('Stanford Car : ', np.array(cars).shape)
print("Car196 Min Year : ", np.min(years))
print("Car196 Max Year : ", np.max(years))
# print(cars)
df = pd.DataFrame(bm).drop_duplicates()
print(df)

''' DVM-CAR '''
file_path = 'DVM-CAR/label/Ad_table.csv'

car_list = pd.read_csv(file_path, encoding='CP949', names=['Maker', 'Genmodel', 'Gen_ID', 'Adv_ID', 'Adv_year', 'Adv_month', 'Color', 'Reg_year', 'Bodytype', 'Runned_Miles', 'Engin_size', 'Gearbox', 'Fuel_type', 'Price', 'Seat_num', 'Door_num'], usecols=['Maker', 'Genmodel', 'Adv_year'], dtype={'Adv_year':str})
car_list = car_list.drop_duplicates(['Maker', 'Genmodel', 'Adv_year'], keep="first")
# car_list = car_list.drop_duplicates(['Maker', 'Genmodel'], keep="first")

dvm_car = []
years = []

for idx in range(len(car_list)):
    car_name = car_list.values[idx]
    years.extend(car_name[-1:])
    dvm_car.append(' '.join(car_name))

years = list(map(int, years))

print('DVM-CAR : ', np.array(dvm_car).shape)
print("DVM-CAR Min Year : ", np.min(years))
print("DVM-CAR Max Year : ", np.max(years))
# print(dvm_car)


''' Comparison '''

equal_cars = []

for idx_d in range(len(dvm_car)):
    for idx_s in range(len(cars)):
        if np.array_equal(dvm_car[idx_d], cars[idx_s]):
            equal_cars.append(dvm_car[idx_d])
            # equal_cars.append(cars[idx_s])
            print("DVM-CAR : ", dvm_car[idx_d], " / Cars196 : ", cars[idx_s])
            break

print("Equal : ", len(equal_cars))
