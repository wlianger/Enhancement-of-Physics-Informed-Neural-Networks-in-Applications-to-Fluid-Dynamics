import csv
from datetime import datetime

from sympy.physics.control.control_plots import matplotlib
from torch.autograd import Variable
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from tools.create_folder import create_folder_if_not_exists
from network_structure.MLP_Xavier import PINN_net
from tools.relative_error import custom_error, REL2
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

current_time = datetime.now()
filename = current_time.strftime("%Y-%m-%d")
result_path = "../result-0.02-decrease/"+filename
create_folder_if_not_exists(result_path)

path_model_init = "../model/u=0.02-decrease/ninit/2024-11-20-0.0050_model-d.pth"
path_model_grad = "../model/u=0.02-decrease/grad/2024-11-20-0.0050_model-d.pth"

read_base_path = "../data/dataset/U=0.02-decrease/"

U = 0.02
L = 0.01

# 统一设置字体
plt.rcParams["font.family"] = 'Times New Roman'

# 分别设置mathtext公式的正体和斜体字体
matplotlib.rcParams['mathtext.fontset'] = 'custom'
matplotlib.rcParams['mathtext.rm'] = 'Times New Roman'  # 用于正常数学文本
matplotlib.rcParams['mathtext.it'] = 'Times New Roman:italic'  # 用于斜体数学文本

fig3 = plt.figure(4, figsize=(10, 3))
fig3.subplots_adjust(hspace=0.6)

af1 = fig3.add_subplot(111)
data = pd.read_csv(read_base_path+"BF_x=0.06.csv")
col_11 = data["x"]
x1 = np.array(col_11)
col_21 = data["y"]
y1 = np.array(col_21)
col_31 = data["u"]
u1 = np.array(col_31)

pt_x1 = Variable(torch.from_numpy(x1).float(), requires_grad=True).unsqueeze(-1)
pt_y1 = Variable(torch.from_numpy(y1).float(), requires_grad=True).unsqueeze(-1)
net_init = torch.load(path_model_init).to("cpu")
net_grad = torch.load(path_model_grad).to("cpu")
in_x = (pt_x1/L)
in_y = (pt_y1/L)
result_init = net_init(torch.cat([in_x, in_y], 1))
result_grad = net_grad(torch.cat([in_x, in_y], 1))

re_u_init = result_init[:, 0].unsqueeze(-1)
re_v_init = result_init[:, 1].unsqueeze(-1)

re_u_grad = result_grad[:, 0].unsqueeze(-1)
re_v_grad = result_grad[:, 1].unsqueeze(-1)

x = np.expand_dims(x1, axis=1)
y = np.expand_dims(y1, axis=1)
u = np.expand_dims(u1, axis=1)

print(len(re_u_init.detach().numpy()))
error_init = custom_error(re_u_init.detach().numpy(), u/U)
error_grad = custom_error(re_u_grad.detach().numpy(), u/U)

print(re_u_grad)
print(re_u_init)
print(error_init)
print(error_grad)

plt.ylabel("y", fontsize=16)
plt.xlabel("error", fontsize=16)
plt.plot(error_init, y / L, "blue", label = "PINN")
plt.plot(error_grad, y / L,  "red",label = "GBPINN")
plt.legend()
plt.show()


# result = np.concatenate((y/L, error_init), axis=1)
# headers = ['y', 'error_init']
# with open('../result-0.02-decrease/PINN-x=02.csv', 'w', newline='') as csvfile:
#     writer = csv.writer(csvfile)
#     writer.writerow(headers)
#     for row in result:
#         writer.writerow(row)
# print("已保存")
#
# result = np.concatenate((y/L, error_grad), axis=1)
# headers = ['y', 'error_grad']
# with open('../result-0.02-decrease/GB-PINN-x=02.csv', 'w', newline='') as csvfile:
#     writer = csv.writer(csvfile)
#     writer.writerow(headers)
#     for row in result:
#         writer.writerow(row)
# print("已保存")

