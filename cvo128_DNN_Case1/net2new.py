import torch
import torch.nn as nn
import numpy as np
import time
import os

class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(10, 50)  # 输入维度 10，隐藏层维度 50
        self.fc2 = nn.Linear(50, 1)   # 输出维度 1

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

if __name__ == "__main__" :

    state_dict = torch.jit.load("submission.pt",map_location='cpu')
    # Xmu0 = state_dict['data_in_mean']
    # Xstd0 = state_dict['data_in_std']
    # Ymu0 = state_dict['data_target_mean']
    # Ystd0 = state_dict['data_target_std']

    # model = state_dict['net']

    print(state_dict)

    #model = SimpleNet()

    # libtorch_model = torch.jit.script(model)

    # libtorch_model.save("new_Temporary_Chemical.pt");

    print("OK")
