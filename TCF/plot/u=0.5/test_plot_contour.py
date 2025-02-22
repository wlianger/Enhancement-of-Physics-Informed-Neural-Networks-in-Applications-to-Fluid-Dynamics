import csv
from datetime import datetime
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from scipy import interpolate
from torch.autograd import Variable
from network_structure.MLP import PINN_net
from tools.create_folder import create_folder_if_not_exists
from tools.relative_error import safe_relative_error, symmetric_relative_error, custom_error, REL2, error_one
import os
# from Wall_shear_graph import global_res4


os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
path_model =  "../../model/u=0.5/0.0510/2024-12-05-0.0510-grad_model"
path_data = "../../data/dataset/u=0.5/TC_verify.csv"

current_time = datetime.now()
filename = current_time.strftime("%Y-%m-%d")
result_path = "../../result/u=0.5/"+filename+"/0.0510/"
create_folder_if_not_exists(result_path)

U = 0.025
L = 0.05
pou = 998.2

data = pd.read_csv(path_data)
col_1 = data["x"]
x = np.array(col_1)
col_2 = data["y"]
y = np.array(col_2)
col_3 = data["av"]
uz = np.array(col_3)
col_4 = data["rv"]
ur = np.array(col_4)
col_5 = data["sv"]
us = np.array(col_5)
col_6 = data["p"]
p = np.array(col_6)
col_7 = data["tv"]
tv = np.array(col_6)

pt_z = Variable(torch.from_numpy(x).float(), requires_grad=True).unsqueeze(-1)
pt_r = Variable(torch.from_numpy(y).float(), requires_grad=True).unsqueeze(-1)

net = PINN_net(2, 256, 4, 7)
net.load_state_dict(torch.load(path_model))
x_d, y_d = np.mgrid[0.0:0.12:50j, 0.05:0.056:50j]

pre_result = net(torch.cat([pt_z/L, pt_r/L], 1))
headers = ['x', 'y', 'p', 'tv', 'uz', 'ur', 'us']

re_uz = pre_result[:, 0]*U
re_us = pre_result[:, 1]*U
re_ur = pre_result[:, 2]*U
re_p = pre_result[:, 3]*U*U*998.2

sum_of_squares = torch.square(re_uz) + torch.square(re_us) + torch.square(re_ur)
pre_tv1 = torch.sqrt(sum_of_squares)

result = torch.cat([pt_z, pt_r, re_p.unsqueeze(-1), pre_tv1.unsqueeze(-1), re_uz.unsqueeze(-1), re_ur.unsqueeze(-1), re_us.unsqueeze(-1)], 1)
result = result.detach().numpy()

with open(result_path+"0510-init.csv", 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(headers)
    for row in result:
        writer.writerow(row)
print("预测结果已保存")

re_uz = re_uz.cpu().detach().numpy()
re_us = re_us.cpu().detach().numpy()
re_ur = re_ur.cpu().detach().numpy()
re_p = re_p.cpu().detach().numpy()

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

def contour(name, rmin, rmax, Spacing_size, pre_d, real_d, error):
    a1 = abs(rmin)
    a2 = abs(rmax)
    error_max = max(a1, a2)
    levels1 = np.arange(rmin, rmax, Spacing_size)
    levels2 = np.arange(0, 0.5, 0.001)
    fig = plt.figure(1, figsize=(13, 7))
    fig.suptitle(name, size=20)
    fig.subplots_adjust(hspace=0.7)

    ax1 = fig.add_subplot(311)
    pre_U = interpolate.griddata((x, y), pre_d, (x_d, y_d), method='cubic')
    pre_cu1 = plt.contourf(x_d, y_d, pre_U, cmap="jet", levels=levels1)
    plt.title("PINN", size=15)
    plt.xlim(0.00, 0.12)
    plt.ylim(0.05, 0.056)
    plt.xlabel("X(m)", size=13)
    plt.ylabel("Y(m)", size=13)
    ax1.tick_params(labelsize=13)
    plt.colorbar(pre_cu1)

    ax2 = fig.add_subplot(312)
    U = interpolate.griddata((x, y), real_d, (x_d, y_d), method='cubic')
    cu1 = plt.contourf(x_d, y_d, U, cmap="jet", levels=levels1)
    plt.title("CFD", size=15)
    plt.xlim(0.00, 0.12)
    plt.ylim(0.05, 0.056)
    plt.xlabel("X(m)", size=13)
    plt.ylabel("Y(m)", size=13)
    ax1.tick_params(labelsize=13)
    fig.colorbar(cu1)

    ax3 = fig.add_subplot(313)
    error_U = interpolate.griddata((x, y), error, (x_d, y_d), method='cubic')
    error_cu1 = plt.contourf(x_d, y_d, error_U, cmap="jet")
    plt.title("Error", size=15)
    plt.xlim(0.00, 0.12)
    plt.ylim(0.05, 0.056)
    plt.xlabel("X(m)", size=13)
    plt.ylabel("Y(m)", size=13)
    ax1.tick_params(labelsize=13)
    fig.colorbar(error_cu1)

    fig.savefig(result_path+name + ".png", bbox_inches='tight', dpi=150, pad_inches=0.1)
    plt.show()

combined = np.hstack((uz/U, ur/U,us/U, p/U/U/pou))
combined_1 = np.hstack((re_uz/U, re_ur/U, re_us/U, re_p/U/U/pou))
error_tu = REL2(combined, combined_1)
print("总误差", error_tu)
# error_uz1 = REL2(uz/U, re_uz/U)
# print("uz误差",error_uz1)
# error_ur1 = REL2(ur/U, re_ur/U)
# print("ur误差",error_ur1)
# error_us1 = REL2(us/U, re_us/U)
# print("us误差",error_us1)
# error_p1 = REL2(p/U/U/pou, re_p/U/U/pou)
# print("p误差",error_p1)

# num = '0.0510'
# number_loss = [num, error_tu, error_uz1, error_ur1, error_us1, error_p1, global_res4]
# with open('../../result/grad_result.csv', 'a', newline='') as csvfile:
#     writer = csv.writer(csvfile)
#     writer.writerow(number_loss)
# print("完成")

# error_uz = custom_error(uz, re_uz)
# error_ur = custom_error(ur, re_ur)
# error_us = custom_error(us, re_us)
# error_p = custom_error(p, re_p)
# sum_of_squares = np.square(re_uz) + np.square(re_us) + np.square(re_ur)
# pre_tv = np.sqrt(sum_of_squares)
# error_tv = custom_error(tv, pre_tv)

x = np.expand_dims(x, axis=1)
y = np.expand_dims(y, axis=1)
uz = np.expand_dims(uz, axis=1)
ur = np.expand_dims(ur, axis=1)
us = np.expand_dims(us, axis=1)
p = np.expand_dims(p, axis=1)

real_value = np.hstack((uz/U, us/U, ur/U, p/pou/U/U))

error_grad = error_one(real_value, pre_result.detach().cpu().numpy())
print(x)
print(y)
print(error_grad)

# x1 = np.expand_dims(x, axis=1)
# y1 = np.expand_dims(y, axis=1)
# error_uz1 = np.expand_dims(error_uz, axis=1)
# error_ur1 = np.expand_dims(error_ur, axis=1)
# error_us1 = np.expand_dims(error_us, axis=1)
# error_p1 = np.expand_dims(error_p, axis=1)
# error_tv1 = np.expand_dims(error_tv, axis=1)
#
# result = np.concatenate((x, y, error_grad), axis=1)
# headers = ['x', 'y', 'error',]
# with open(result_path+"0510-init-error.csv", 'w', newline='') as csvfile:
#     writer = csv.writer(csvfile)
#     writer.writerow(headers)
#     for row in result:
#         writer.writerow(row)
# print("预测误差已保存")

# contour("Axial_Velocity", -5*(1e-3),  5*(1e-3), 1e-4, re_uz, uz, error_uz)
# contour("Radial_Velocity", -5*(1e-3), 6.0*(1e-3), 1e-4, re_ur, ur, error_ur)
# contour("Swirl_Velocity", 0.0, 0.055, 0.001, re_us, us, error_us)
# contour("Static_Pressure", -5.5*(1e-2), 5.8*(1e-2), 1e-3, re_p, p, error_p)











