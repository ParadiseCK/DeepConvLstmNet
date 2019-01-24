import torch
from pre_data.generate_person_data import *
import numpy as np
import matplotlib.pyplot as plt
def get_init_data ():
    all_coor = []
    for i in range(26):
        COOR = []
        for j in range(85):
            a = []
            id_table = joblib.load("./pre_data/box_data/2/PersonId/"+str(j)+".id")
            coordinate = get_person_xy(i + 1, id_table)
            line  = coordinate[0]
            row = coordinate[1]
            a.append(20-line)
            a.append(row)
            COOR.append(np.array(a))
        all_coor.append(np.array(COOR))
    # print(np.array(all_coor).shape)
    return np.array(all_coor)
def get_result(label_data, coordinate):
    real_C = []
    base_line = coordinate[0]
    balse_row = coordinate[1]
    for i in range (label_data.shape[0]):
        data = label_data[i][0]
        max_index = np.where(data == np.max(data))
        dalt_line = max_index[0][0] - 1
        dalt_row = max_index[1][0]  -1
        base_line = base_line + dalt_line
        balse_row = balse_row + dalt_row
        real_C.append([20-base_line,balse_row ])
    real_label = np.array(real_C)
    return real_label
color = ['#FFA500',
        '#FAEBD7',
        '#00FFFF',
        '#7FFFD4',
        '#F0FFFF',
        '#F5F5DC',
        '#FFE4C4',
        '#000000',
        '#FFEBCD',
        '#0000FF',
        '#8A2BE2',
        '#A52A2A',
        '#DEB887',
        '#5F9EA0',
        '#7FFF00',
        '#D2691E',
        '#FF7F50',
        '#6495ED',
        '#B0E0E6',
        '#DC143C',
        '#8B4513',
        '#00008B',
        '#6A5ACD',
        '#B8860B',
        '#D2B48C',
        '#9ACD32']
file_name = "2/"
all_real_lable = []
all_predict_lable = []
init_data = get_init_data ()
for i in range (26):
    label_data_path = "./pre_data/train_label/" + file_name + str(i + 1) + ".label"
    label_data = joblib.load(label_data_path)
    id_table = joblib.load("./pre_data/box_data/"+file_name+"PersonId/84.id")
    coordinate = get_person_xy(i+1, id_table)
    real_lable = get_result(label_data, coordinate)
    all_real_lable.append(real_lable)
for i , data in enumerate(all_real_lable):
    new_data = np.concatenate((init_data[i], data), axis=0)
    plt.figure(1)
    plt.title("Real Label")
    plt.plot(new_data[:, 1], new_data[:, 0], c=color[i])
    plt.scatter(new_data[:, 1], new_data[:, 0], c=color[i])
    plt.xlim(0, 25)
    plt.ylim(0, 20)
plt.show()