#coding:utf-8
import csv
import numpy as np
import matplotlib.pyplot as plt
from pre_data.frame_data import *
from pre_data.generate_person_data import *
from Conv_LSTM import *
from torch.autograd import Variable
from train import Pedestrian_CLSTM
from options import *

def to_float(data):
    DATA = []
    if data.strip() != '':
        Y = float(data.split(',')[1:][0])
        X = float(data.split(',')[:1][0])
        DATA.append(X)
        DATA.append(Y)
    return DATA
def read_csv_init(csv_path, inint_frame_long):
    video = []
    for i in range(inint_frame_long+1):
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
def plot_csv(csv_path, color):
    with open(csv_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for i ,line in enumerate(reader):
            Line = []
            for data in line:
                new_data = data[1:len(data)-1]
                Line.append(to_float(new_data))
            plt.figure(1)
            plt.plot(np.array(Line)[:,0],np.array(Line)[:,1], c=color[i])
        plt.show()
def Generate_inint_feat(init_video, base_x_y, person_box, full_box, object_set, boundary, terminal):
    video = init_video
    Person_Id = get_Person_id(video, full_box, person_box, base_x_y)
    X_t_1 = get_X_1(video, full_box, person_box, base_x_y)
    X_t_2 = get_X_2(video, full_box, person_box, base_x_y)
    X_t_3 = get_X_3(video, full_box, person_box, base_x_y, object_set, boundary)
    X_t_4 = get_X_4(video, full_box, person_box, base_x_y)
    X_t_5 = get_X_5(video, full_box, person_box, base_x_y, terminal)
    X_t_6 = get_X_6(video, full_box, person_box, base_x_y, terminal)
    Feat = []
    for i in range(np.array(video).shape[0]):
        frame_feat = []
        frame_feat.append(X_t_1[i])
        frame_feat.append(X_t_2[i])
        frame_feat.append(X_t_3[i])
        frame_feat.append(X_t_4[i])
        frame_feat.append(X_t_5[i])
        frame_feat.append(X_t_6[i])
        Feat.append(np.array(frame_feat))
    return Feat, Person_Id
def generate_person_data_init(Feat, PersonId, person_num):
    all_person_data = []
    for id in range(person_num):
        PERSON_DATA = []
        for i , data in enumerate(Feat):
            person_data = get_person_data_per(data, PersonId[i], id+1)
            if person_data is not None:
                PERSON_DATA.append(person_data)
        all_person_data.append(np.array(PERSON_DATA))
    return all_person_data
def generate_train_inint_data(person_data, seq_len, data_gap):
    Data_X = []
    data_x = []
    for i in range(seq_len):
        x = person_data[(i * data_gap)]
        data_x.append(x)
    if len(data_x) > 0:
        Data_X.append(np.array(data_x))
    return np.array(Data_X, dtype="float32")

def generate_inint_scene(init_video, person_num):
    base_x_y = opts['base_x_y']
    person_box = opts['person_box']  # 一个人所占格子大小单位：米
    full_box = opts['full_box']  #4-3 将全局分为30*36的方格子
    boundary = opts['boundary']  # [上边界宽度，左边界宽度， 下边界宽度]
    object_set = opts['object_set']
    terminal = opts['terminal']  # 目标点坐标
    Feat, PersonId = Generate_inint_feat(init_video, base_x_y,
                                         person_box, full_box, object_set, boundary, terminal)  # 生成特征文件并保存

    all_person_data = generate_person_data_init(Feat, PersonId, person_num)  # 生成每一位行人的训练数据

    Input_data = []
    for data in all_person_data:
        train_data = generate_train_inint_data(data, seq_len, data_gap)
        Input_data.append(train_data[0])
    return np.array(Input_data)

def to_arry(video):
    arr_data = []
    for data in video:
        new_data = to_float(data)
        arr_data.append(new_data)
    # print(np.array(arr_data).shape)
    return np.array(arr_data)

def to_str(arry):
    List = []
    del_person =[]
    for i, data in enumerate(arry):
        if data[0] < 13:
            List.append(str(data[0]) + "," + str(data[1]))
        else:
            List.append(str(data[0]) + "," + str(data[1]))
            del_person.append(i)
    del_person.sort()
    return List, del_person
if __name__ == "__main__":
    csv_path = "./data_csv/4-3-f.csv"
    vedio_long = np.array(read_csv(csv_path)).shape[0]
    filename = os.path.split(csv_path)[1].split(".")[0]
    Result_path = "./result/"
    if not os.path.isdir(Result_path):
        os.makedirs(Result_path)
    init_video = None
    seq_len = opts['seq_len']
    data_gap = opts['data_gap']
    inint_frame_long = seq_len  * data_gap
    VIDEO = read_csv_init(csv_path, inint_frame_long)
    Person_Path = []
    print (np.array(VIDEO).shape)
    print (vedio_long-(opts["seq_len"] * opts["data_gap"]))
    for frame in range(1000):
    # for frame in range(vedio_long):
        if frame == 0:
            init_video = read_csv_init(csv_path, inint_frame_long)
        person_num = len(init_video[0])
        input_data = generate_inint_scene(init_video, person_num)
        num_features = [32, 64, 128]
        filter_size = 3
        shape = (5, 5)
        inp_chans = 6
        nlayers = 3
        seq_len = 8
        model_path = opts['model_path']
        batch_size = person_num
        train_x = torch.from_numpy(input_data)
        input = Variable(train_x).cuda()
        conv_lstm = CLSTM(shape, inp_chans, filter_size, num_features, nlayers)
        p_conv_lstm = Pedestrian_CLSTM(shape, inp_chans, filter_size, num_features, nlayers)
        p_conv_lstm.load_state_dict(torch.load(model_path))
        p_conv_lstm.cuda()
        hidden_state = conv_lstm.init_hidden(batch_size)
        out = p_conv_lstm(input, hidden_state)
        predict = out.cpu().data.numpy()
        # print(init_video[-1])
        # print(predict.tolist())
        # print(to_arry(init_video[-1]).shape)
        # print(predict.shape)
        next_out = to_arry(init_video[-1]) + predict*(1/30)

        _next_out , del_person= to_str(next_out)
        del init_video[0]
        init_video.append(_next_out)
        VIDEO.append(_next_out)

        # pesonlist_ = []
        # for data in init_video:
        #     pesonlist_.append(len(data))
        # print(pesonlist_)
        if len (del_person) > 0:
            for i, p in enumerate(del_person):
                for k, data1 in enumerate(init_video):
                    init_video[k] = np.delete(np.array(data1),(p-i)).tolist()

        if len (del_person) > 0:
            print (del_person)
            for i ,p in enumerate(del_person):
                # print (p)
                per_path = []
                for k, data2 in enumerate(VIDEO):
                    per_path.append(data2[p-i])
                    VIDEO[k] = np.delete(np.array(data2), (p-i)).tolist()
                Person_Path.append(per_path)

        # pesonlist =[]
        # for data in init_video:
        #     pesonlist.append(len(data))
        # print(pesonlist)
        print ("*"*10 + str(frame) + "*"*10)
        print(np.min(next_out[:, 0]))
        print (len(VIDEO[0]))
        if np.min(next_out[:, 0]) > opts['over']:
            break
    if len (VIDEO)>0:
        for i in range(len(VIDEO[0])):
            per_path = []
            for _frame_ in VIDEO:
                per_path.append(_frame_[i])
            Person_Path.append(per_path)

    # finall_result_path = Result_path+filename+"_v"+str(os.path.split(opts["model_path"])[1].split(".")[0][-1])+".path"
    # # finall_result_path = Result_path + filename + "_v10.result"
    # joblib.dump(Person_Path, finall_result_path)

    video_ = read_csv_init(csv_path,1)
    # print (video_[0])

    first_per = []
    for path in Person_Path:
        first_per.append(path[0])
    first_per_ = np.array(first_per)


    Finall_path =[]
    for data in video_[0]:
        index = np.where(first_per_ == data)
        index_ = index[0].tolist()
        if len (index_) > 0:
            Finall_path.append(Person_Path[index_[0]])


    result_csv = "./result_csv/" + filename + "_v.csv"
    if not os.path.isdir("./result_csv/"):
        os.makedirs("./result_csv/")
    with open(result_csv, "w") as csvfile:
        writer = csv.writer(csvfile)
        for data in Finall_path:
            writer.writerow(data)