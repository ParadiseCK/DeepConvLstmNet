#coding:utf-8
import numpy as np
import os
from sklearn.externals import joblib

def get_person_xy(id, personid):
    for i, line in enumerate(personid):
        # print(str(i) + "line:{}".format(line))
        for j, row in enumerate(line):
            # print(str(i) + "," + str(j) + "data:{}".format(row))
            if str(id) in row.split(","):
                return (i,j)#(line_index, row_index)

def get_person_data_per(frame, id_table, person_id):
    coordinate = get_person_xy(person_id, id_table)
    if coordinate is not None:
        person_data_list = []
        for i in range(frame.shape[0]):
            frame_data_per = frame[i]
            person_data_per = np.zeros((5,5))
            for line in range(5):
                for row in range(5):
                    _line_ = coordinate[0]- 2 + line
                    _row_ = coordinate[1] -2 + row
                    if (-1<_line_ < frame.shape[1]) and (-1<=_row_<frame.shape[2]):
                        person_data_per[line, row] = frame_data_per[_line_,_row_]
                    else:
                        person_data_per[line, row] = 0
            person_data_list.append(person_data_per)
        return np.array(person_data_list)
    else:
        return None



def generate_person_data(base_feat_path, base_id_path, person_data_basepath, person_num):
    feat_list = sorted([p for p in os.listdir(base_feat_path) if os.path.splitext(p)[1] == ".feat"])
    name_list = []
    for feat_name in feat_list:
        name_list.append(int(os.path.split(feat_name)[1].split(".")[0]))
    name_list_ = np.array(name_list)
    name_list.sort()
    for id in range(person_num):
        PERSON_DATA = []
        for i in name_list:
            finall_feat_path = base_feat_path + str(i) +".feat"
            finall_id_path = base_id_path + str(i) + ".id"
            feat = joblib.load(finall_feat_path)
            id_table = joblib.load(finall_id_path)
            person_data = get_person_data_per(feat, id_table, id+1)
            if person_data is not None:
                PERSON_DATA.append(person_data)
        # print(np.array(PERSON_DATA).shape)
        if not os.path.isdir(person_data_basepath):
            os.makedirs(person_data_basepath)
        data_name = str(id+1)+".person"
        person_data_path = os.path.join(person_data_basepath, data_name)
        joblib.dump(np.array(PERSON_DATA), person_data_path)
        print (str(id+1)+"号行人的数据:"+str(np.array(PERSON_DATA).shape)+"已注入")