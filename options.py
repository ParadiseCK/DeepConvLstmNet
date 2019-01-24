from collections import OrderedDict

opts = OrderedDict()



opts['feat_path'] = "./box_data/"
opts['person_path'] = "./person_data/"
opts['base_x_y'] = [0, 0]
opts['person_box'] = [0.4, 0.4]  # 一个人所占格子大小单位：米
opts['full_box'] = [30, 36]  # 将全局分为30*36的方格子
opts['boundary'] = [5, 9, 9]  # [上边界宽度，左边界宽度， 下边界宽度]
# opts['object_set'] = ([9.308, 7.481, 3, 2], [12, 10, 6, 1], [12, 6.3, 7, 1])  # [左上角横坐标，左上角纵坐标， 障碍物在新的格子中占3行2列]
# opts['object_set'] = ([10.231, 7.481, 3, 2], [12, 10, 6, 1], [12, 6.3, 7, 1])  # [左上角横坐标，左上角纵坐标， 障碍物在新的格子中占3行2列]
opts['object_set']  = ([9.724, 7.234, 2, 3], [12, 10, 6, 1], [12, 6.3, 7, 1])  # [左上角横坐标，左上角纵坐标， 障碍物在新的格子中占3行2列]
opts['terminal'] = [14, 6.8]#目标点坐标
opts['seq_len'] = 8
opts['data_gap'] = 8
opts['over'] = 12.5
opts['model_path'] = "../model/model_finall.pth"

