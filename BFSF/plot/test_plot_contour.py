import csv
from datetime import datetime
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from scipy import interpolate
from sympy.physics.control.control_plots import matplotlib
from torch.autograd import Variable
from network_structure.MLP_Xavier import PINN_net
from tools.create_folder import create_folder_if_not_exists
from tools.relative_error import safe_relative_error, custom_error, REL2, error_one
import os
# from Wall_shear_graph import global_res4
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

path_model = "../model/u=0.02-decrease/grad/2024-11-20-0.0050_model-d.pth"
path_data = "../data/dataset/U=0.02-decrease/BF_verify.csv"

current_time = datetime.now()
filename = current_time.strftime("%Y-%m-%d")
result_path = "../result-0.02/"+filename+"/-0.0080/"
create_folder_if_not_exists(result_path)
U = 0.02
L = 0.01
pou = 998.2

data = pd.read_csv(path_data)
col_1 = data["x"]
x = np.array(col_1)
col_2 = data["y"]
y = np.array(col_2)
col_3 = data["u"]
u = np.array(col_3)
col_4 = data["v"]
v = np.array(col_4)
col_5 = data["p"]
p = np.array(col_5)
col_5 = data["tv"]
tv = np.array(col_5)

pt_x = Variable(torch.from_numpy(x).float(), requires_grad=True).unsqueeze(-1)
pt_y = Variable(torch.from_numpy(y).float(), requires_grad=True).unsqueeze(-1)
pt_u = Variable(torch.from_numpy(u).float(), requires_grad=True).unsqueeze(-1)

# net =PINN_net(2, 128, 3, 6)
# net.load_state_dict(torch.load(path_model))
net = torch.load(path_model).to("cpu")
x_d, y_d = np.mgrid[-0.089:0.0:100j, 0.0:0.010:30j]
x_d1, y_d1 = np.mgrid[0.0:0.178:100j, -0.010:0.010:30j]

# for name, param in net.named_parameters():
    # print(f"Parameter name: {name}")
    # print(f"Parameter shape: {param.shape}")
    # print(f"Parameter values:\n{param.data}\n")

input_x = pt_x/L
input_y = pt_y/L
pre_result = net(torch.cat([input_x,input_y], 1))
# print("预测结果",pre_result)
re_u1 = pre_result[:, 0]*U
re_v1 = pre_result[:, 1]*U
re_p1 = pre_result[:, 2]*pou*U*U

sum_of_squares = torch.square(re_u1) + torch.square(re_v1)
pre_tv1 = torch.sqrt(sum_of_squares)

re_u = re_u1.cpu().detach().numpy()
re_v = re_v1.cpu().detach().numpy()
re_p = re_p1.cpu().detach().numpy()
pre_tv = pre_tv1.cpu().detach().numpy()

result = torch.cat([pt_x, pt_y, re_u1.unsqueeze(-1), re_v1.unsqueeze(-1), re_p1.unsqueeze(-1), pre_tv1.unsqueeze(-1)], 1)
result = result.detach().numpy()

# headers = ['x', 'y', 'u', 'v', 'p', 'Velocity Magnitude']
# with open('../result-0.02-decrease/0050-init.csv','w',newline='') as csvfile:
#     writer = csv.writer(csvfile)
#     writer.writerow(headers)
#     for row in result:
#         writer.writerow(row)

# 统一设置字体
plt.rcParams["font.family"] = 'Times New Roman'

# 分别设置mathtext公式的正体和斜体字体
matplotlib.rcParams['mathtext.fontset'] = 'custom'
matplotlib.rcParams['mathtext.rm'] = 'Times New Roman'  # 用于正常数学文本
matplotlib.rcParams['mathtext.it'] = 'Times New Roman:italic'  # 用于斜体数学文本

def contour(name, rmin, rmax, Spacing_size, pre_d, real_d, error, level2):
    levels1 = np.arange(rmin, rmax, Spacing_size)
    levels2 = np.arange(-0.001, level2, 0.001)
    fig = plt.figure(1, figsize=(17, 10))
    fig.suptitle(name, size=20)
    fig.subplots_adjust(hspace=0.7)

    ax1 = fig.add_subplot(311)
    pre_U = interpolate.griddata((x, y), pre_d, (x_d, y_d), method='cubic')
    pre_U1 = interpolate.griddata((x, y), pre_d, (x_d1, y_d1), method='cubic')
    pre_cu1 = plt.contourf(x_d, y_d, pre_U, cmap="jet", levels=levels1)
    pre_cu2 = plt.contourf(x_d1, y_d1, pre_U1, cmap="jet", levels=levels1)
    plt.title("PINN", size=20)
    plt.xlim(-0.089, 0.178)
    plt.ylim(-0.01, 0.01)
    plt.xlabel(r"$\it{x}$$\rm(m)$", fontsize=20)
    plt.ylabel(r"$\it{y}$$\rm(m)$", fontsize=20)
    ax1.tick_params(labelsize=20)
    plt.colorbar(pre_cu2)
    ax2 = fig.add_subplot(312)
    U = interpolate.griddata((x, y), real_d, (x_d, y_d), method='cubic')
    U1 = interpolate.griddata((x, y), real_d, (x_d1, y_d1), method='cubic')
    cu1 = plt.contourf(x_d, y_d, U, cmap="jet", levels=levels1)
    cu2 = plt.contourf(x_d1, y_d1, U1, cmap="jet", levels=levels1)
    plt.title("CFD", size=20)
    plt.xlim(-0.089, 0.178)
    plt.ylim(-0.01, 0.01)
    plt.xlabel(r"$\it{x}$$\rm(m)$", fontsize=20)
    plt.ylabel(r"$\it{y}$$\rm(m)$", fontsize=20)
    ax2.tick_params(labelsize=20)
    fig.colorbar(cu2)

    ax3 = fig.add_subplot(313)
    error_U = interpolate.griddata((x, y), error, (x_d, y_d), method='cubic')
    error_U1 = interpolate.griddata((x, y), error, (x_d1, y_d1), method='cubic')
    error_cu1 = plt.contourf(x_d, y_d, error_U, cmap="jet",  levels=levels2)
    error_cu2 = plt.contourf(x_d1, y_d1, error_U1, cmap="jet",  levels=levels2)
    plt.title("Error", size=20)
    plt.xlim(-0.089, 0.178)
    plt.ylim(-0.01, 0.01)
    plt.xlabel(r"$\it{x}$$\rm(m)$", fontsize=20)
    plt.ylabel(r"$\it{y}$$\rm(m)$", fontsize=20)
    ax3.tick_params(labelsize=20)
    fig.colorbar(error_cu2)
    fig.savefig(result_path+name + ".png", bbox_inches='tight', dpi=600, pad_inches=0.1)
    plt.show()

combined = np.hstack((u/0.02,v/0.02,p/0.02/0.02/998.2))
combined_1 = np.hstack((re_u/0.02, re_v/0.02, re_p/0.02/0.02/998.2))
res = REL2(combined,combined_1)
print("整体l2误差:")
print(res)
res1 = REL2(u/0.02, re_u/0.02)
print("ul2误差:")
print(res1)
res2 = REL2(v/0.02, re_v/0.02)
print("vl2误差:")
print(res2)
res3 = REL2(p/0.02/0.02/998.2, re_p/0.02/0.02/998.2)
print("p误差:")
print(res3)

x = np.expand_dims(x, axis=1)
y = np.expand_dims(y, axis=1)
u = np.expand_dims(u, axis=1)
v = np.expand_dims(v, axis=1)
p = np.expand_dims(p, axis=1)

real_value = np.hstack((u/U,v/U,p/(pou*U*U)))
error_grad = error_one(real_value, pre_result.detach().cpu().numpy())


result = np.concatenate((x/L, y/L, error_grad), axis=1)
headers = ['x', 'y', 'error']
with open('../result-0.02-decrease/0050-grad-error.csv','w',newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(headers)
    for row in result:
        writer.writerow(row)

print("已保存")


# contour("X_Velocity", -0.002, 0.03, 1e-4, re_u, u, error_x, 0.2)
#
# contour("Y_Velocity", -2.6*1e-3, 2.4*1e-3, 1e-5, re_v, v, error_y, 1.0)
#
# contour("Static_Pressure", -9.8*1e-2, 0.38, 1e-3, re_p, p, error_p, 0.05)
#
# contour("Total_Velocity", -0.0001, 0.028, 1e-4, pre_tv, tv, error_tv, 0.2)







