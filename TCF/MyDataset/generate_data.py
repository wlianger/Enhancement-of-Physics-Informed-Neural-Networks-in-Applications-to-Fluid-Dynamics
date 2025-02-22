import numpy as np
import pandas as pd
import torch


if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")


def boundary_data(num_samples, domain):
    bc_zeros_lr = torch.zeros((100, 1))
    bc_zeros_lr_500 = torch.zeros((500, 1))

    z_bc_interval1 = domain[0] + (domain[1] - domain[0]) * torch.rand(500, 1)  # 0.0-0.12
    z_bc_value = torch.ones((100, 1)) * domain[1]  # 0.12

    r_bc_interval1 = domain[2] + (domain[3] - domain[2]) * torch.rand(100, 1)  # 0.05-0.056
    r_bc_value1 = torch.ones((500, 1))*domain[2]  # 0.05
    r_bc_value2 = torch.ones((500, 1))*domain[3]  # 0.056

    bc_data1 = torch.cat([bc_zeros_lr, r_bc_interval1], 1).to(device)  # z=0.00; r(0.05，0.056)
    bc_data2 = torch.cat([z_bc_value, r_bc_interval1], 1).to(device)  # z=0.12; r(0.05，0.056)
    bc_data3 = torch.cat([z_bc_interval1, r_bc_value1], 1).to(device)  # z(0.0,0.12); r=0.05
    bc_data4 = torch.cat([z_bc_interval1, r_bc_value2], 1).to(device)  # z(0.0,0.12); r=0.056
    bc_data5 = torch.cat([bc_zeros_lr_500, r_bc_value1], 1).to(device)  # z=0; r=0.05

    return bc_data1, bc_data2, bc_data3, bc_data4, bc_data5


def equation_data(num_samples, domain):
    z_collocation = domain[0] + (domain[1] - domain[0]) * torch.rand(num_samples, 1, requires_grad=True)  # 0.0-0.12
    r_collocation = domain[2] + (domain[3] - domain[2]) * torch.rand(num_samples, 1, requires_grad=True)  # 0.05-0.056
    f_data = torch.cat([z_collocation, r_collocation], 1).to(device)
    # print(f_data)
    return f_data


def supervision_data(path, u, l):
    U = u
    L = l
    rou = 998.2

    data = pd.read_csv(path)

    col_1 = data["x"]
    x = np.array(col_1)
    col_2 = data["y"]
    y = np.array(col_2)
    col_3 = data["av"]
    uz = np.array(col_3)
    col_4 = data["rv"]
    ur= np.array(col_4)
    col_5 = data["sv"]
    us = np.array(col_5)
    col_6 = data["p"]
    p = np.array(col_6)

    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)
    uz = uz.reshape(-1, 1)
    us = us.reshape(-1, 1)
    ur = ur.reshape(-1, 1)
    p = p.reshape(-1, 1)

    x = torch.from_numpy(x).to(device)
    x = (x / L).to(torch.float32)
    y = torch.from_numpy(y).to(device)
    y = (y / L).to(torch.float32)
    uz = torch.from_numpy(uz).to(device)
    uz = (uz / U).to(torch.float32)
    ur = torch .from_numpy(ur).to(device)
    ur = (ur / U).to(torch.float32)
    us = torch.from_numpy(us).to(device)
    us = (us / U).to(torch.float32)
    p = torch.from_numpy(p).to(device)
    p = (p / (rou*U*U)).to(torch.float32)

    input_data = torch.cat([x, y], 1)
    real_data = torch.cat([uz, us, ur, p], 1)

    # print(input_data, real_data)
    return input_data, real_data







