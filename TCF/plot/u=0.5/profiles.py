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
from tools.relative_error import safe_relative_error, symmetric_relative_error, REL2, error_one
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

current_time = datetime.now()
filename = current_time.strftime("%Y-%m-%d")
result_path = "../../result/u=0.5/"+filename+"/0.0510/"
create_folder_if_not_exists(result_path)
path_model_grad = "../../model/u=0.5/0.0510/2024-12-05-0.0510-grad_model"
path_model_init = "../../model/u=0.5/0.0510/2024-12-05-0.0510_model"
read_base_path = "../../data/dataset/u=0.5/"

U = 0.025
L = 0.05

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']


data2 = pd.read_csv(read_base_path+"TC_z_0.06_all.csv")
col_12 = data2["x"]
x2 = np.array(col_12)
col_22 = data2["y"]
y2 = np.array(col_22)
col_32 = data2["sv"]
d2 = np.array(col_32)
pt_x2 = Variable(torch.from_numpy(x2).float(), requires_grad=True).unsqueeze(-1)
pt_y2 = Variable(torch.from_numpy(y2).float(), requires_grad=True).unsqueeze(-1)
net = PINN_net(2, 256, 4, 7)
net.load_state_dict(torch.load(path_model_init))
result2 = net(torch.cat([pt_x2/L, pt_y2/L], 1))
re_s = result2[:, 1].unsqueeze(-1)


x = np.expand_dims(d2, axis=1)
print(d2)
print(re_s.detach().numpy())
res_error = error_one(x/U, re_s.detach().numpy())



plt.plot(res_error, (y2-0.05)/0.006, label='GBPINN')
# plt.plot(y2, d2, label='Data')
plt.show()

# result = torch.cat([pt_x2/0.006, (pt_y2-0.05)/0.006, re_s],dim=1)
# result = result.detach().numpy()

# headers = ['x', 'y', 'value']
#
# with open(result_path+"0.06-swirl-velocity.csv", 'w', newline='') as csvfile:
#     writer = csv.writer(csvfile)
#     writer.writerow(headers)
#     for row in result:
#         writer.writerow(row)
# print("预测结果已保存")

# result = torch.cat([pt_x2/0.006, (pt_y2-0.05)/0.006, re_s],dim=1)
# result = result.detach().numpy()
# headers = ['x', 'y', 'error']
# with open(result_path+"0.06-swirl-velocity.csv", 'w', newline='') as csvfile:
#     writer = csv.writer(csvfile)
#     writer.writerow(headers)
#     for row in result:
#         writer.writerow(row)
# print("预测结果已保存")

