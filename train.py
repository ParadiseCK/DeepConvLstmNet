from Conv_LSTM import *
from torch import nn, optim
from torch.autograd import Variable
from sklearn.externals import joblib
import torch.nn.functional as F
import os
import random
import numpy as np

class Pedestrian_CLSTM(nn.Module):
    def __init__(self, shape, input_chans, filter_size, num_features, num_layers):
        super(Pedestrian_CLSTM, self).__init__()
        self.shape = shape  # H,W
        self.input_chans = input_chans
        self.filter_size = filter_size
        self.num_features = num_features
        self.num_layers = num_layers

        self.layer1 = CLSTM(self.shape, self.input_chans, self.filter_size, self.num_features, self.num_layers)
        self.layer2 =nn.Conv2d(self.num_features[-1], (self.num_features[-1])*2, kernel_size=3, padding=0, stride=1)
        self.fc1 = nn.Linear((self.num_features[-1])*2*3*3, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 2)

    def forward(self, input, hidden_state):
        hidden_out = self.layer1(input, hidden_state)
        out = F.relu(self.layer2(hidden_out[1][-1]))
        out = out.view(-1, (self.num_features[-1])*2*3*3)
        out = F.dropout(out, 0.5)
        out = F.relu(self.fc1(out))
        out = F.dropout(out, 0.5)
        out = F.relu(self.fc2(out))
        out = F.dropout(out, 0.5)
        out = self.fc3(out)
        return out
if __name__ == "__main__":
    num_features = [32, 64, 128]
    filter_size = 3
    shape = (5, 5)  # H,W
    inp_chans = 6
    nlayers = 3
    seq_len = 8
    sequence_data = ["slow_1/","slow_2/","fast_1/","fast_2/",
                     "1/","1-1/","2-1/","2-1-f/","2-2/","2-2-f/","3-1/",
                     "3-1-f/","3-2/","3-2-f/","4-1/","4-1-f/","4-2/",
                     "4-2-f/","5-1/","5-1-f/","5-2/","5-2-f/"]

    model_path = "./model/"
    if not os.path.isdir(model_path):
        os.makedirs(model_path)

    conv_lstm = CLSTM(shape, inp_chans, filter_size, num_features, nlayers)
    p_conv_lstm = Pedestrian_CLSTM(shape, inp_chans, filter_size, num_features, nlayers)
    print('convlstm module:', p_conv_lstm)
    p_conv_lstm.apply(weights_init)
    # p_conv_lstm.load_state_dict(torch.load(model_path + "model_1.pth"))
    p_conv_lstm.cuda()
    criterion = nn.MSELoss()
    loss = None
    total_epoch = 1300
    learning_rate = 1e-3
    for epoch in range(total_epoch):
        optimizer = optim.Adam(p_conv_lstm.parameters(), lr=learning_rate)
        random_sequence = random.sample(sequence_data, len(sequence_data))
        if (epoch + 1) % 1 == 0:
            print("=" * 200)
            print("train sequence:{}".format(random_sequence))
            print("=" * 200)
        LOSS = []
        for file_name in random_sequence:
            arr = np.arange(26)
            np.random.shuffle(arr)
            for i in arr:
                train_data_path = "./train_data-8/"+file_name + str(i + 1) + ".train"
                label_data_path = "./train_label-8/"+file_name + str(i + 1) + ".label"
                if os.path.exists(train_data_path) and os.path.exists(label_data_path):
                    train_data = joblib.load(train_data_path)
                    # print(train_data.shape)
                    label_data = joblib.load(label_data_path)
                    # print(label_data.shape)
                    batch_size = train_data.shape[0]

                    train_x = torch.from_numpy(train_data)
                    train_y = torch.from_numpy(label_data)

                    input = Variable(train_x).cuda()
                    lable = Variable(train_y).cuda()

                    hidden_state = conv_lstm.init_hidden(batch_size)
                    out = p_conv_lstm(input, hidden_state)
                    loss = criterion(out, lable)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    if (epoch + 1) % 1 == 0:
                        print('epoch:{} ,scene: {},person: {}, loss is: {}'.format(str(epoch+1),file_name[0:len(file_name)-1],i + 1, loss.data[0]))
                    loss = loss.data[0]
                    LOSS.append(loss)
        if (epoch + 1) > 299 and (epoch + 1) < 600:
            learning_rate = 1e-4
        if (epoch + 1) > 599 and (epoch + 1) <900:
            learning_rate = 1e-5
        if (epoch + 1) > 899 and (epoch + 1) <1100:
            learning_rate = 1e-6
        if (epoch + 1) > 1099:
            learning_rate = 1e-7
        if (epoch + 1) % 100 == 0:
            torch.save(p_conv_lstm.state_dict(), model_path+"model_"+str(learning_rate)+"_"+str(round(np.mean(np.array(LOSS)),10))+"_"+str(epoch+1)+".pth")
            print ("Saved The Model")
