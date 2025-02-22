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
result_path = "../../result/u=0.5/"+filename
create_folder_if_not_exists(result_path)
path_model = "../../model/u=0.5/0.0510/2024-12-06-0.0510-grad_model"
read_base_path = "../../data/dataset/u=0.5/"

U = 0.025
L = 0.05

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']


data2 = pd.read_csv(read_base_path+"TC_r_bottom.csv")
col_12 = data2["x"]
x2 = np.array(col_12)
col_22 = data2["y"]
y2 = np.array(col_22)
col_32 = data2["dswirl-velocity-dy"]
d2 = np.array(col_32)
pt_x2 = Variable(torch.from_numpy(x2).float(), requires_grad=True).unsqueeze(-1)
pt_y2 = Variable(torch.from_numpy(y2).float(), requires_grad=True).unsqueeze(-1)
net = PINN_net(2, 256, 4, 7)
net.load_state_dict(torch.load(path_model))
result2 = net(torch.cat([pt_x2/L, pt_y2/L], 1))
re_s = result2[:, 1]*U

result_us_r = torch.autograd.grad(re_s, pt_y2, create_graph=True, grad_outputs=torch.ones_like(re_s))[0]


plt.plot(x2, result_us_r[:, 0].detach().numpy()*0.006/0.025, label='GBPINN')
plt.plot(x2, d2*0.006/0.025, label='Data')
plt.show()

error = REL2(d2*0.006/0.025,  result_us_r[:, 0].detach().numpy()*0.006/0.025)
print("剪切力误差：",error)

result = torch.cat([pt_x2/0.006, (pt_y2-0.05)/0.006, result_us_r[:, 0].unsqueeze(-1)*0.006/0.025],dim=1)
result = result.detach().numpy()

# headers = ['x', 'y', 'grad']
#
# with open(result_path+"gradient-05012-grad.csv", 'w', newline='') as csvfile:
#     writer = csv.writer(csvfile)
#     writer.writerow(headers)
#     for row in result:
#         writer.writerow(row)
# print("预测结果已保存")
