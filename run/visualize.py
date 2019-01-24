#coding:utf-8
import csv
import numpy as np
import matplotlib.pyplot as plt
from pre_data.frame_data import *
from sklearn.externals import joblib

def read_result_csv(csv_path):
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
                        data = line[i]
                        frame.append(data)
                    else :
                        frame.append("")
        video.append(frame)
    return video

def to_float(data):
    DATA = []
    if data.strip() != '':
        Y = float(data.split(',')[1:][0])
        X = float(data.split(',')[:1][0])
        DATA.append(X)
        DATA.append(Y)
    return DATA
def plot_compair (csv_path,result_csv_path, color, box, boundary):
    with open(csv_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        index = -1
        for i, line in enumerate(reader):
            if len(line) > (opts["seq_len"] * opts["data_gap"] + 1):
                index = index+1
                Line = []
                # if index !=28:
                for data in line:
                    new_data = data[1:len(data)-1]
                    Line.append(to_float(new_data))
                plt.figure(1)
                plt.title("Real")
                plt.plot(np.array(Line)[:,0],np.array(Line)[:,1], c=color[index])
                plt.plot(box[:, 0], box[:, 1], c="r", linewidth="2")
                plt.plot(np.array(boundary)[:, 0], np.array(boundary)[:, 1], c="r", linewidth="2")
                plt.xlim(3, 14)
                plt.ylim(3, 11)
    with open(result_csv_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        index_ = -1
        for i, line in enumerate(reader):
            index_ = index_+1
            Line = []
            # if index !=28:
            for data in line:
                Line.append(to_float(data))
            plt.figure(2)
            plt.title("Predict")
            plt.plot(np.array(Line)[:,0],np.array(Line)[:,1], c=color[index_])
            plt.plot(box[:, 0], box[:, 1], c="r", linewidth="2")
            plt.plot(np.array(boundary)[:, 0], np.array(boundary)[:, 1], c="r", linewidth="2")
            plt.xlim(3, 14)
            plt.ylim(3, 11)
    plt.show()

def plot_dynamic(video, box, color, boundary, img_path):
    plt.ion()
    plt.figure(1)

    for index, frame in enumerate(video):
        plt.clf()
        Frame = []
        for  data in frame:
            if data.strip() != '':
                Frame.append(to_float(data))
            else :
                Frame.append(-1)
        for i, lable in enumerate(Frame):
            if lable != -1 and lable[0] < 12:
                plt.scatter(lable[0],lable[1], c=color[i],marker=">", s=100)
                plt.plot(box[:, 0], box[:, 1], c="r", linewidth="2")
                plt.plot(np.array(boundary)[:, 0], np.array(boundary)[:, 1], c="r", linewidth="2")
                plt.xlim(3, 14)
                plt.ylim(3, 11)

        plt.pause(0.0001)
        plt.show()

        # if not os.path.isdir(img_path):
        #     os.makedirs(img_path)
        # plt.savefig(img_path + str(index) + ".png")


def plot_per_frame(video, result, box, boundary):
    plt.ion()
    All_x_error = []
    All_y_error = []
    for i in range(0,len(video)):
        plt.clf()
        real= video[i]
        predict = result[i]
        PREDICT = []
        REAL = []
        for data in real:
            if data.strip() != '':
                REAL.append(to_float(data))
            else :
                REAL.append(-1)
        for i, lable in enumerate(REAL):
            if lable != -1:
                plt.scatter(lable[0],lable[1], c=color[i], s=100)
                plt.plot(box[:, 0], box[:, 1], c="r", linewidth="2")
                plt.plot(np.array(boundary)[:, 0], np.array(boundary)[:, 1], c="r", linewidth="2")
                plt.xlim(3, 14)
                plt.ylim(3, 11)
        for data in predict:
            if data.strip() != '':
                PREDICT.append(to_float(data))
            else :
                PREDICT.append(-1)
        for i, lable in enumerate(PREDICT):
            if lable != -1:
                plt.scatter(lable[0],lable[1], c=color[i], marker=">", s=120)
                plt.plot(box[:, 0], box[:, 1], c="r", linewidth="2")
                plt.plot(np.array(boundary)[:, 0], np.array(boundary)[:, 1], c="r", linewidth="2")
                plt.xlim(3, 14)
                plt.ylim(3, 11)
        # ERROR = []
        # if len(REAL)>0:
        #     for i ,data in enumerate(REAL):
        #         if data is not -1:
        #             error = np.array(PREDICT[i]) - np.array(REAL[i])
        #             ERROR.append(error)
        #     x_error = np.mean(abs(np.array(ERROR)[:,0]))
        #     y_error = np.mean(abs(np.array(ERROR)[:, 1]))
        #     All_x_error.append(x_error)
        #     All_y_error.append(y_error)
        #     print ("横向位置误差均值：{}".format(x_error))
        #     print("纵向位置误差均值：{}".format(y_error))

        plt.pause(0.0001)

        plt.show()

    print("场景横向位置误差均值均值：{}".format(np.mean(np.array(All_x_error))))
    print("场景纵向位置误差均值均值：{}".format(np.mean(np.array(All_y_error))))

def object(base_x_y, hight, width):
    point1 = base_x_y,
    point2 = [base_x_y[0]+width, base_x_y[1]]
    point3 = [base_x_y[0]+width, base_x_y[1]-hight]
    point4 = [base_x_y[0], base_x_y[1]-hight]
    rect = [point1[0],point2,point3,point4,point1[0]]
    return np.array(rect)


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

if __name__ == "__main__":
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
             '#9ACD32',
             '#ADFF2F',
             '#F0FFF0',
             '#F0E68C',
             '#FFA07A',
             '#20B2AA',
             '#778899',
             '#800000']
    csv_path = "./data_csv/4-3-f.csv"
    filename = os.path.split(csv_path)[1].split(".")[0]
    result_path = "./result_csv/" + str(filename) + "_v.csv"
    video = read_csv(csv_path)
    result = read_result_csv(result_path)
    # box = object([9.308, 7.55], 1.2, 0.6)
    # box = object([10.0, 7.55], 1.2, 0.6)
    box = object([9.724, 7.4], 0.6, 1.2)
    boundary = [[12, 6.3], [12, 3.7], [3.7, 3.7], [3.7, 10.5], [12, 10.5], [12, 7.7]]
    person_num = np.array(video).shape[1]
    # plot_csv(csv_path, color, boundary, box)

    # plot_result(result,color)
    plot_compair(csv_path, result_path,color, box, boundary)
    img_path = "./"+filename+"-img-move/"
    # plot_dynamic(result, box, color, boundary, img_path)
    # plot_per_frame(video, result,box ,boundary)





    # Person_Path = joblib.load("./result/1-1-1_vl.path")
    # video_ = read_csv_init(csv_path, 1)
    # print (video_[0])
    #
    # first_per = []
    # for path in Person_Path:
    #     print (path[0])
    #     first_per.append(path[0])
    # first_per_ = np.array(first_per)
    # Finall_path = []
    # for data in video_[0]:
    #     index = np.where(first_per_ == data)
    #     index_ = index[0].tolist()
    #     if len(index_) > 0:
    #         Finall_path.append(Person_Path[index_[0]])
    #     else:
    #         print("hahahahahahahha")
    # filename = os.path.split(csv_path)[1].split(".")[0]
    # result_csv = "./result_csv/" + filename + "_v1.csv"
    # if not os.path.isdir("./result_csv/"):
    #     os.makedirs("./result_csv/")
    # with open(result_csv, "w") as csvfile:
    #     writer = csv.writer(csvfile)
    #     for data in Finall_path:
    #         writer.writerow(data)
