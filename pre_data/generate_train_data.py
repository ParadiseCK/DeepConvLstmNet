#coding:utf-8
import numpy as np
import os
from sklearn.externals import joblib

def generate_train_data(person_data, lable_data, seq_len, data_gap):
    # print (lable_data.shape)
    # print (person_data.shape)
    Data_X = []
    Data_Y = []
    for index in range(person_data.shape[0]):
        data_x = []
        data_y = []
        for i in range(seq_len):
            if (index + ((seq_len-1)* data_gap)) < (person_data.shape[0]-1):
                x = person_data[index + (i * data_gap)]
                data_x.append(x)
        if len(data_x) > 0:
            # print(len(lable_data))
            # print (seq_len*data_gap+index)
            if seq_len*data_gap+index < len(lable_data):
                Data_X.append(np.array(data_x))
                data_y.append(lable_data[(seq_len*data_gap+index)])
                Data_Y.append(np.array(data_y))
    # print (np.array(Data_X).shape)
    # print (np.array(Data_Y).shape)
    return np.array(Data_X, dtype="float32"), np.array(Data_Y, dtype="float32")


# if __name__ == "__main__":
#     filename = "1/"
#     person_data_path = "./person_data/"+filename
#     lable_data_path = "./person_label/"+filename
#     train_in_path = "./train_data/"+filename
#     train_out_path = "./train_label/"+filename
#     seq_len = 8
#     data_gap = 12
#     person_num = 26
#     if not os.path.isdir(train_in_path):
#         os.makedirs(train_in_path)
#     if not os.path.isdir(train_out_path):
#         os.makedirs(train_out_path)
#     for i in range(person_num):
#         person_data = joblib.load(person_data_path + str(i+1)+".person")
#         lable_data = joblib.load(lable_data_path + str(i+1)+".label")
#         train_data, train_label = generate_train_data(person_data, lable_data, seq_len, data_gap)
#         train_data_path = train_in_path+str(i+1)+".train"
#         train_label_path = train_out_path+str(i+1)+".label"
#         joblib.dump(train_data, train_data_path)
#         joblib.dump(train_label, train_label_path)
#         print(str(i + 1) + "号行人的train数据:" + str(train_data.shape) + "已生成")
#         print(str(i + 1) + "号行人的label数据:" + str(train_label.shape) + "已生成")
#         print ("*"*50)
