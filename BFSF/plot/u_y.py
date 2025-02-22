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
result_path = "../result/"+filename+"/-0.0092/"
create_folder_if_not_exists(result_path)

path_model = "../model/u=0.02-decrease/ninit/2024-11-21-0.0085_model.pth"
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
data = pd.read_csv(read_base_path+"BF_bottom.csv")
col_11 = data["x"]
x1 = np.array(col_11)
col_21 = data["y"]
y1 = np.array(col_21)
col_31 = data["dx-velocity-dy"]
d = np.array(col_31)

pt_x1 = Variable(torch.from_numpy(x1).float(), requires_grad=True).unsqueeze(-1)
pt_y1 = Variable(torch.from_numpy(y1).float(), requires_grad=True).unsqueeze(-1)
# net =PINN_net(2, 128, 3, 7)
# net.load_state_dict(torch.load(path_model))
# net.eval()
net = torch.load(path_model).to("cpu")
in_x = (pt_x1/L)
in_y = (pt_y1/L)
result1 = net(torch.cat([in_x, in_y], 1))
re_u1 = result1[:, 0].unsqueeze(-1)
re_v1 = result1[:, 1].unsqueeze(-1)
f = torch.autograd.grad(re_u1, in_y, grad_outputs=torch.ones_like(re_u1), create_graph=True)[0]

plt.title("x-wall-shear y=bottom", size=20)
plt.xlabel(r"$\it{x}$", fontsize=20)
plt.ylabel(r"$\it{value}$", fontsize=20)
plt.plot(x1/L, f[:,0].detach().numpy(), "y-", label="GB-PINN", linewidth=3)
plt.plot(x1/L, d/U*L, "r-", label="CFD", linewidth=3)
print(x1)
plt.xlim(0.0, 17.8)
af1.tick_params(labelsize=20)
plt.legend()

global_res4 = REL2(d*0.001003, 0.001003*f[:,0].detach().numpy()*U/L)
print("l2相对误差")
print(global_res4)

x = np.expand_dims(x1, axis=1)
y = np.expand_dims(y1, axis=1)
u_y = np.expand_dims(d/U*L, axis=1)
pre_u_y = np.expand_dims(f[:,0].detach().numpy(), axis=1)

result = np.concatenate((x/L, y/L, u_y, pre_u_y), axis=1)
headers = ['x', 'y', 'CFD', 'GB-PINN']
with open('../result-0.02-decrease/0050_wall_shell_init.csv','w',newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(headers)
    for row in result:
        writer.writerow(row)

print("已保存")


# plt.axvline(0, color='black', linestyle='--')
#
# af1 = fig3.add_subplot(212)
# error = custom_error(0.001003*f[:,0].detach().numpy()*U/L,d*0.001003)
# plt.title("error", size=20)
# plt.xlabel("x(m)", size=20)
# plt.ylabel("error", size=20)
# plt.plot(x1, error, "r-", label="PINN", linewidth=2)
# plt.xlim(0.089, 0.267)
# af1.tick_params(labelsize=20)
plt.show()
# fig3.savefig(result_path+"wall-shear-stress-0.0092" + ".png", bbox_inches='tight', dpi=300, pad_inches=0.1)
