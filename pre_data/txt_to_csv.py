import csv
import numpy as np
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
                if i < len(line):
                    data = line[i][1:len(line[i])-1]
                    frame.append(data)
                else :
                    frame.append("")
        video.append(frame)
    return video
base_path = "/home/chenkai/对应标号/采集数据1/纵向障碍物-窄门-无奖励-3(5-3)"
with open("./data/5-3.csv","w") as csvfile:
    writer = csv.writer(csvfile)
    for i in range(26):
        filename = str(i+1)
        a = np.loadtxt(base_path+"/"+filename+".txt")
        A = []
        for data in a:
            A.append(data.tolist())
        writer.writerow(A)
# video = read_csv("./test.csv")
# print (len(video[0]))
# print (np.array(video).shape)