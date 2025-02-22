import csv
from datetime import datetime

import numpy
from torch.autograd import Variable
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from network_structure.MLP import PINN_net
from tools.create_folder import create_folder_if_not_exists
from tools.relative_error import safe_relative_error, symmetric_relative_error, REL2
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

current_time = datetime.now()
filename = current_time.strftime("%Y-%m-%d")
result_path = "../../result/u=0.5/"+filename+"/0.0510/"
create_folder_if_not_exists(result_path)
path_model = "../../model/u=0.5/0.0510/2024-12-05-0.0510-grad_model"
read_base_path = "../../data/dataset/u=0.5/"

U = 0.025
L = 0.05

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']


data2 = pd.read_csv(read_base_path+"TC_r_bottom.csv")
col_12 = data2["x"]
x2 = np.array(col_12, dtype=float)
col_22 = data2["y"]
y2 = np.array(col_22, dtype=float)
col_32 = data2["dswirl-velocity-dy"]
d2 = np.array(col_32, dtype=float)
d2 = d2/0.025*0.006
pt_x2 = Variable(torch.from_numpy(x2).float(), requires_grad=True).unsqueeze(-1)
pt_y2 = Variable(torch.from_numpy(y2).float(), requires_grad=True).unsqueeze(-1)
net = PINN_net(2, 256, 4, 7)
net.load_state_dict(torch.load(path_model))
result2 = net(torch.cat([pt_x2/L, pt_y2/L], 1))
re_s = result2[:, 1].unsqueeze(-1)
f2 = torch.autograd.grad(re_s, pt_y2, grad_outputs=torch.ones_like(re_s), create_graph=True)[0]
f2 = f2.detach().numpy()*0.006
def simpson_rule(y, h):
    """
    y -- 数列，表示函数在等间距点上的取值
    h -- 相邻点之间的距离
    """
    n = len(y) - 1
    if n % 2 != 0:
        raise ValueError("数列长度减一不是偶数")
    integral = y[0] + y[-1]
    for i in range(1, n):
        if i % 2 == 0:
            integral += 2 * y[i]
        else:
            integral += 4 * y[i]
    return integral
print(simpson_rule(f2, 0.12/800)/2400*2/149.3)

