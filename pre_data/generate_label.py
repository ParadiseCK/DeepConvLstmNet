import numpy as np
from sklearn.externals import joblib
from pre_data.generate_person_data import *

def get_x_y_v (coordinate, feat):
    data = []
    x_v = feat[0][coordinate]
    y_v = feat[1][coordinate]
    data.append(x_v)
    data.append(y_v)
    return np.array(data)
def generate_labe(base_id_path,base_feat_path, lable_path, person_num):
    id_list = sorted([p for p in os.listdir(base_id_path) if os.path.splitext(p)[1] == ".id"])
    name_list = []
    for feat_name in id_list:
        name_list.append(int(os.path.split(feat_name)[1].split(".")[0]))
    name_list.sort()
    for id in range(person_num):
        PERSON_id=[]
        LABLE = []
        PERSON_feat = []
        for i in name_list:
            finall_id_path = base_id_path + str(i) + ".id"
            finall_feat_path = base_feat_path + str(i) + ".feat"
            id_table = joblib.load(finall_id_path)
            feat = joblib.load(finall_feat_path)
            coordinate = get_person_xy(id+1, id_table)
            if coordinate is not None:
                PERSON_id.append(coordinate)
                PERSON_feat.append(feat)
        for i in range(len(PERSON_id)):
            if i < (len(PERSON_id)-1):
                lable = get_x_y_v(PERSON_id[i+1], PERSON_feat[i+1])
                LABLE.append(lable)
        if not os.path.isdir(lable_path):
            os.makedirs(lable_path)
        label_name = str(id+1)+".label"
        person_lable_path = os.path.join(lable_path, label_name)
        joblib.dump(np.array(LABLE), person_lable_path)
        print (str(id+1)+"号行人的Label值:"+str(np.array(LABLE).shape)+"已注入")
# person_num = 26
# csv_path = "./data/1.csv"
# feat_path = "./box_data/"
# person_path ="./person_data/"
# filename = os.path.split(csv_path)[1].split(".")[0]
# base_id_path = feat_path+filename+"/PersonId/"#存放行人ID 的文件路径
# base_feat_path = feat_path+filename+"/feat/"#存放每一帧数据的文件路径
# label_path = "./person_label/"+filename+"/"
# generate_labe(base_id_path,base_feat_path,label_path, person_num)