#coding:utf-8
import csv
import numpy as np
import math
import os
from sklearn.externals import joblib
from options import *

# 读取CSV文件按列返回数据
def read_csv(csv_path):
    num = []
    video = []
    with open(csv_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            num.append(len(line))
    colume_num = max(num)
    for i in range(colume_num):
        frame = []
        with open(csv_path, 'r') as csvfile:
            reader = csv.reader(csvfile)
            for line in reader:
                if len(line) > (opts["seq_len"] * opts["data_gap"] + 1):
                    if i < len(line):
                        data = line[i][1:len(line[i])-1]
                        frame.append(data)
                    else :
                        frame.append("")
        video.append(frame)
    return video
#将数据返回为行人ID、新的坐标和在方格中的索引[person_id, x, y, x_index, y_index]
def change_to_new_factor(frame, full_box, person_box, base_x_y):
    FRAME = []
    for i, data in enumerate(frame):
        new_factor = []
        if data.strip()!='':
            y = float(data.split(',')[1:][0])
            x = float(data.split(',')[:1][0])
            # print("x:{}  y:{}".format(x, y))
            index_x = math.floor((x - base_x_y[0]) / person_box[0])#向上取整
            index_y = math.floor((y - base_x_y[1]) / person_box[1])
            # print ("index_x:{}  index_y:{}".format(index_x,index_y))
            new_factor.append(i+1)
            new_factor.append(x)
            new_factor.append(y)
            new_factor.append((full_box[0] - index_y +1))
            new_factor.append(index_x)
        else:
            new_factor.append(i+1)
            new_factor.append(-1)
            new_factor.append(-1)
            new_factor.append(-1)
            new_factor.append(-1)
        FRAME.append(new_factor)
    return np.array(FRAME)

# 处理再同一个格子的点
def Process_Same_Point(frame):
    x_y = frame[:,3:5]
    x_y_ = []
    for i, data in enumerate(x_y):
        # print ("index:{}→x_y:{}".format(i, data))
        x_y_.append((str(int(data[0]))+"."+str(int(data[1]))))
    # print (x_y_)
    vals, inverse, count = np.unique(x_y_, return_inverse=True,
                                     return_counts=True)
    idx_vals_repeated = np.where(count > 1)[0]

    if len(idx_vals_repeated) < 1:
        return True
    else:
        return False

#将行人的ID标入每一帧他所在的格子中
def get_Person_id(video, full_box, person_box, base_x_y):
    Person_Id = []
    # print (np.array(video).shape)
    for i in range(np.array(video).shape[0]):
        # print("frame: {}".format(i + 1))
        frame_t = np.array(video)[i]
        new_frame_t = change_to_new_factor(frame_t, full_box, person_box, base_x_y)
        # print(new_frame_t)
        person_id = np.zeros((full_box[0],full_box[1]))
        person_id = person_id.astype(np.str)
        for i, data in enumerate(new_frame_t[:,3:5]):
            if data[0] > 0:
                in_data = person_id[(int(data[0]-1), int(data[1]-1))]
                if  in_data.strip()=='0.0':
                    person_id[(int(data[0] - 1), int(data[1] - 1))] = str(i+1)
                else:
                    # print("*" * 10)
                    person_id[(int(data[0] - 1), int(data[1] - 1))] = str(in_data) + "," +str(i+1)
        Person_Id.append(person_id)
    return np.array(Person_Id)
#X_1 表示横向的行人速度
def get_X_1(video, full_box, person_box, base_x_y):
    X_1 = []
    for i in range(np.array(video).shape[0]):
        x_1 = np.zeros((full_box[0], full_box[1]))
        if i == 0:
            # print("frame: {}".format(i + 1))
            X_1.append(x_1)
        else:
            # print("frame: {}".format(i + 1))
            frame_t = np.array(video)[i]
            frame_t_1 = np.array(video)[i-1]
            new_frame_t = change_to_new_factor(frame_t, full_box, person_box, base_x_y)
            new_frame_t_1 = change_to_new_factor(frame_t_1, full_box, person_box, base_x_y)
            # print (new_frame_t)
            # print (new_frame_t_1)
            for i , data in enumerate(new_frame_t):
                if data[1] >0:
                    dalt_x_v = (data[1] - new_frame_t_1[i][1]) * 30
                    in_data = x_1[(int(data[3] - 1), int(data[4] - 1))]
                    if in_data == 0:
                        x_1[(int(data[3] - 1), int(data[4] - 1))] = dalt_x_v
                    else:
                        # print ("*"*10)
                        x_1[(int(data[3] - 1), int(data[4] - 1))] = (dalt_x_v + in_data)/2
            X_1.append(x_1)
    # print (np.array(X_1))
    return np.array(X_1)
#X_2 表示纵向的行人速度
def get_X_2(video, full_box, person_box, base_x_y):
    X_2 = []
    for i in range(np.array(video).shape[0]):
        x_2 = np.zeros((full_box[0], full_box[1]))
        if i == 0:
            # print("frame: {}".format(i + 1))
            X_2.append(x_2)
        else:
            # print("frame: {}".format(i + 1))
            frame_t = np.array(video)[i]
            frame_t_1 = np.array(video)[i-1]
            new_frame_t = change_to_new_factor(frame_t, full_box, person_box, base_x_y)
            new_frame_t_1 = change_to_new_factor(frame_t_1, full_box, person_box, base_x_y)
            # print (new_frame_t)
            # print (new_frame_t_1)
            for i , data in enumerate(new_frame_t):
                if data[1] >0:
                    dalt_x_v = (data[2] - new_frame_t_1[i][2]) * 30
                    in_data = x_2[(int(data[3] - 1), int(data[4] - 1))]
                    if in_data == 0:
                        x_2[(int(data[3] - 1), int(data[4] - 1))] = dalt_x_v
                    else:
                        # print ("*"*10)
                        x_2[(int(data[3] - 1), int(data[4] - 1))] = (dalt_x_v + in_data)/2
            X_2.append(x_2)
    # print (np.array(X_2))
    return np.array(X_2)
#X_3 表示每个格子的可达性(0代表可到达， 1代表不可到达)固定障埃物
def get_X_3(video, full_box, person_box, base_x_y, object_set, boundary):
    X_3 = []
    for i in range(np.array(video).shape[0]):
        frame_t = np.array(video)[i]
        new_frame_t = change_to_new_factor(frame_t, full_box, person_box, base_x_y)
        x_3 = np.zeros((full_box[0], full_box[1]))
        for object in object_set:
            init_object_x = math.ceil((object[0] - base_x_y[0]) / person_box[0])
            init_object_y = full_box[0] - math.ceil((object[1] - base_x_y[1]) / person_box[1]) + 1
            target_object_x = init_object_x + object[3] - 1
            target_object_y = init_object_y + object[2] - 1
            for i in range(init_object_y, target_object_y + 1):
                for j in range(init_object_x, target_object_x + 1):
                    # print("行:{}  列:{}".format(i, j))
                    x_3[(int(i - 1), int(j - 1))] = 1
        for i in range(full_box[0]):
            for j in range(full_box[1]):
                if i < boundary[0]:
                    x_3[(i, j)] = 1
                if i >full_box[0] - boundary[2] - 1:
                    x_3[(i, j)] = 1
                if j < boundary[1]:
                    x_3[(i, j)] = 1
        X_3.append(x_3)
    return np.array(X_3)
#X_4表示每个格子人群密度
def get_X_4(video, full_box, person_box, base_x_y):
    X_4 = []
    for i in range(np.array(video).shape[0]):
        frame_t = np.array(video)[i]
        new_frame_t = change_to_new_factor(frame_t, full_box, person_box, base_x_y)
        x_4 = np.zeros((full_box[0], full_box[1]))
        for i, data in enumerate(new_frame_t[:, 3:5]):
            if data[0] > 0:
                in_data = x_4[(int(data[0] - 1), int(data[1] - 1))]
                if in_data == 0:
                    x_4[(int(data[0] - 1), int(data[1] - 1))] = 1
                else:
                    x_4[(int(data[0] - 1), int(data[1] - 1))] = in_data + 1
        X_4.append(x_4)
    return np.array(X_4)

#X_5 每个格子横向到目标点的倒数
def get_X_5 (video, full_box, person_box, base_x_y, terminal):
    X_5 = []
    T_x = math.floor((terminal[0] - base_x_y[0]) / person_box[0])  # 向上取整
    for i in range(np.array(video).shape[0]):
        frame_t = np.array(video)[i]
        x_5 = np.zeros((full_box[0], full_box[1]))
        for i in range(full_box[0]):
            for j in range( full_box[1]):
                # print("行:{}  列:{}".format(i, j))
                distance = (T_x-j) * person_box[0]
                x_5[(i, j)] = distance/terminal[0]
        X_5.append(x_5)
    return np.array(X_5)
#X_6 每个格子纵向到目标点的倒数
def get_X_6 (video, full_box, person_box, base_x_y, terminal):
    X_6 = []
    T_y = math.floor((terminal[1] - base_x_y[0]) / person_box[0])  # 向上取整
    for i in range(np.array(video).shape[0]):
        x_6 = np.zeros((full_box[0], full_box[1]))
        for i in range(full_box[0]):
            for j in range( full_box[1]):
                # print("行:{}  列:{}".format(i, j))
                if i < (full_box[0] - T_y):
                    distance = -((full_box[0] - T_y)-i) * person_box[1] - (person_box[1]/2)
                else:
                    distance = (i-(full_box[0] - T_y)+1) *person_box[1] - (person_box[1]/2)
                x_6[(i, j)] = distance/terminal[1]
        X_6.append(x_6)
    return np.array(X_6)


def Generate_feat(csv_path, feat_path, base_x_y, person_box, full_box, object_set, boundary, terminal):
    video = read_csv(csv_path)
    print (np.array(video).shape)
    Person_Id = get_Person_id(video, full_box, person_box, base_x_y)
    print("生成格子中行人的ID:{}================》》》Done！！！".format(Person_Id.shape))
    X_t_1 = get_X_1(video, full_box, person_box, base_x_y)
    print("生成横向行人的速度X_1:{}================》》》Done！！！".format(X_t_1.shape))
    X_t_2 = get_X_2(video, full_box, person_box, base_x_y)
    print("生成纵向行人的速度X_2:{}================》》》Done！！！".format(X_t_2.shape))
    X_t_3 = get_X_3(video, full_box, person_box, base_x_y, object_set , boundary)
    print("生成每个格子的可达性X_3:{}================》》》Done！！！".format(X_t_3.shape))
    X_t_4 = get_X_4(video, full_box, person_box, base_x_y)
    print("生成每个格子的人群密度X_4:{}================》》》Done！！！".format(X_t_4.shape))
    X_t_5 = get_X_5 (video, full_box, person_box, base_x_y, terminal)
    print("生成每个格子横向到目标点距离X_5:{}================》》》Done！！！".format(X_t_5.shape))
    X_t_6 = get_X_6(video, full_box, person_box, base_x_y, terminal)
    print("生成每个格子纵向到目标点距离X_6:{}================》》》Done！！！".format(X_t_6.shape))
    for i in range(np.array(video).shape[0]):
        frame_feat = []
        frame_feat.append(X_t_1[i])
        frame_feat.append(X_t_2[i])
        frame_feat.append(X_t_3[i])
        frame_feat.append(X_t_4[i])
        frame_feat.append(X_t_5[i])
        frame_feat.append(X_t_6[i])
        base_name = os.path.split(csv_path)[1].split(".")[0]
        feat_name = str(i) + ".feat"
        Id_name =str(i) + ".id"
        new_feat_path = feat_path + base_name +"/feat"
        new_ID_path = feat_path + base_name + "/PersonId"
        if not os.path.isdir(new_feat_path):
            os.makedirs(new_feat_path)
        if not os.path.isdir(new_ID_path):
            os.makedirs(new_ID_path)
        fd_path = os.path.join(new_feat_path, feat_name)
        Id_path = os.path.join(new_ID_path, Id_name)
        _frame_feat_ = np.array(frame_feat)
        joblib.dump(_frame_feat_, fd_path)
        joblib.dump(Person_Id[i], Id_path)
    print ("数据保存完成！！！")
# if __name__ == "__main__":
    # csv_path = "./data/1.csv"
    # feat_path = "./box_data/"
    # base_x_y = [0, 0]
    # person_box = [0.4, 0.4]#一个人所占格子大小单位：米
    # full_box = [30, 36] #将全局分为30*36的方格子
    # boundary = [5, 9, 9]#[上边界宽度，左边界宽度， 下边界宽度]
    # object_set = ([10.231, 7.481, 3, 2], [12 ,10, 6,1], [12, 6.3, 7,1])#[左上角横坐标，左上角纵坐标， 障碍物在新的格子中占3行2列]
    # terminal = [14, 6.8]
    # Generate_feat(csv_path, feat_path, base_x_y, person_box, full_box, object_set, boundary, terminal)#生成特征文件并保存
#
#
#     video = read_csv(csv_path)
#     X_t_3 = def get_X_3(video, full_box, person_box, base_x_y, object_set, boundary)
#     print (X_t_3[0].shape)
#     for i in range(X_t_3[0].shape[0]):
#         print (X_t_3[0][i,:].tolist())

    # feat = joblib.load(".\\box_data\\横向障碍物-宽门-无奖励-1\\feat\\1.feat")
    # print (feat.shape)
    # id = joblib.load(".\\box_data\\横向障碍物-宽门-无奖励-1\\PersonId\\1.id")
    # print(id)
    #
    # # 按序读取特征文件
    # base_feat_path = ".\\box_data\\横向障碍物-宽门-无奖励-1\\feat\\"
    # feat_list = sorted([p for p in os.listdir(base_feat_path) if os.path.splitext(p)[1] == ".feat"])
    # name_list = []
    # for feat_name in feat_list:
    #     name_list.append(int(os.path.split(feat_name)[1].split(".")[0]))
    # name_list_ = np.array(name_list)
    # name_list.sort()
    # for i in name_list:
    #     finall_path = base_feat_path + str(i) +".feat"
    #     joblib.load(finall_path)
    # feat = joblib.load("./box_data/1-1/feat/1.feat")
    # DATA=[]
    # for i in range(feat[5].shape[0]):
    #     data1 = feat[4][i,:].tolist()
    #     data2 = feat[5][i,:].tolist()
    #     data = []
    #     for i in range(len(data1)):
    #         d= str(round(data1[i], 2))+","+str(round(data2[i], 2))
    #         data.append(d)
    #     DATA.append(data)
    # with open("./destination.csv", "w") as csvfile:
    #     writer = csv.writer(csvfile)
    #     for data in DATA:
    #         writer.writerow(data)



