#coding:utf-8
from pre_data.frame_data import *
from pre_data.generate_person_data import *
from pre_data.generate_label import *
from pre_data.generate_train_data import *
from options import *

if __name__ == "__main__":
    csv_path = "./data/5-2-f.csv"
    feat_path = opts['feat_path']
    person_path = opts['person_path']
    base_x_y = opts['base_x_y']
    person_box = opts['person_box']  # 一个人所占格子大小单位：米
    full_box = opts['full_box']  # 将全局分为30*36的方格子
    boundary = opts['boundary']  # [上边界宽度，左边界宽度， 下边界宽度]
    object_set = opts['object_set']  # [左上角横坐标，左上角纵坐标， 障碍物在新的格子中占3行2列]
    terminal =opts['terminal']#目标点坐标
    Generate_feat(csv_path, feat_path, base_x_y, person_box, full_box, object_set, boundary, terminal)  # 生成特征文件并保存
    print ("="*50)
    print ("="*20 + "生成行人数据"+"="*20)
    print("=" * 50)

    filename = os.path.split(csv_path)[1].split(".")[0]
    base_feat_path = feat_path+filename+"/feat/"#存放每一帧数据的文件路径
    base_id_path = feat_path+filename+"/PersonId/"#存放行人ID 的文件路径
    person_data_basepath = person_path+filename+"/"#存放行人在场景中的数据的路径

    person_num = np.array(read_csv(csv_path)).shape[1]

    generate_person_data(base_feat_path, base_id_path, person_data_basepath, person_num)#生成每一位行人的训练数据

    print("=" * 50)
    print("="*17 +"生成行人数据 Lable"+"="*17)
    print("=" * 50)
    label_path = "./person_label/"+filename+"/"
    generate_labe(base_id_path,base_feat_path, label_path, person_num)




    #生成训练数据
    person_data_path = "./person_data/" + filename + "/"
    lable_data_path = "./person_label/" + filename + "/"
    train_in_path = "./train_data/" + filename + "/"
    train_out_path = "./train_label/" + filename + "/"
    seq_len = opts['seq_len']
    data_gap = opts['data_gap']
    if not os.path.isdir(train_in_path):
        os.makedirs(train_in_path)
    if not os.path.isdir(train_out_path):
        os.makedirs(train_out_path)
    for i in range(person_num):
        person_data = joblib.load(person_data_path + str(i + 1) + ".person")
        lable_data = joblib.load(lable_data_path + str(i + 1) + ".label")
        train_data, train_label = generate_train_data(person_data, lable_data, seq_len, data_gap)
        train_data_path = train_in_path + str(i + 1) + ".train"
        train_label_path = train_out_path + str(i + 1) + ".label"
        joblib.dump(train_data, train_data_path)
        joblib.dump(train_label, train_label_path)
        print(str(i + 1) + "号行人的train数据:" + str(train_data.shape) + "已生成")
        print(str(i + 1) + "号行人的label数据:" + str(train_label.shape) + "已生成")
        print("*" * 50)