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
pou = 998.2
UL = 0.006

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']


data2 = pd.read_csv(read_base_path+"TC_r_0.053_all.csv")
col_12 = data2["x"]
x2 = np.array(col_12)
col_22 = data2["y"]
y2 = np.array(col_22)
col_32 = data2["av"]
av = np.array(col_32)
col_42 = data2["rv"]
rv = np.array(col_42)
col_52 = data2["sv"]
sv = np.array(col_52)
col_62 = data2["p"]
p = np.array(col_62)

pt_x2 = Variable(torch.from_numpy(x2).float(), requires_grad=True).unsqueeze(-1)
pt_y2 = Variable(torch.from_numpy(y2).float(), requires_grad=True).unsqueeze(-1)
net = PINN_net(2, 256, 4, 7)
net.load_state_dict(torch.load(path_model_init))
result_init = net(torch.cat([pt_x2 / L, pt_y2 / L], 1))

net = PINN_net(2, 256, 4, 7)
net.load_state_dict(torch.load(path_model_grad))
result_grad = net(torch.cat([pt_x2 / L, pt_y2 / L], 1))

x = np.expand_dims(x2, axis=1)
y = np.expand_dims(y2, axis=1)
av = np.expand_dims(av, axis=1)
rv = np.expand_dims(rv, axis=1)
sv = np.expand_dims(sv, axis=1)
p = np.expand_dims(p, axis=1)

real_value = np.hstack([av/U, sv/U, rv/U, p/(pou*U*U)])

res_error_init = error_one(real_value, result_init.detach().numpy())
res_error_grad = error_one(real_value, result_grad.detach().numpy())

# plt.plot(res_error_init, (y2-0.05)/UL, label='PINN')
# plt.plot(res_error_grad, (y2-0.05)/UL, label='GBPINN')
# plt.show()

plt.plot(x, res_error_init, label='PINN')
plt.plot(x, res_error_grad, label='GBPINN')
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
#
result = numpy.hstack([x/UL, (y-0.05)/UL, res_error_init])
headers = ['x', 'y', 'error']
with open(result_path+"r_0.5-error_init.csv", 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(headers)
    for row in result:
        writer.writerow(row)
print("预测结果已保存")

result = numpy.hstack([x/UL, (y-0.05)/UL, res_error_grad])
headers = ['x', 'y', 'error']
with open(result_path+"r_0.5-error_grad.csv", 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(headers)
    for row in result:
        writer.writerow(row)
print("预测结果已保存")

