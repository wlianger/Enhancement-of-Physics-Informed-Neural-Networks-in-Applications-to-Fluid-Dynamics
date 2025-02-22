import numpy as np
import pandas as pd
import torch

SEED = 1
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")


def boundary_data(num_samples, domain):
    bc_zeros = torch.zeros((166, 1))
    bc_zeros_1 = torch.zeros((50, 1))

    x_bc_interval1 = domain[0] + (domain[1] - domain[0]) * torch.rand(166, 1)  # 0.0-8.9  #-8.9到0.0
    x_bc_interval2 = domain[0] + (domain[2] - domain[0]) * torch.rand(500, 1)  # 0.0-26.7 #-8.9到17.8
    x_bc_interval3 = domain[1] + (domain[2] - domain[1]) * torch.rand(334, 1)  # 8.9-26.7 #0.0-17.8
    x_bc_value = torch.ones((50, 1)) * domain[0]  # 8.9 #-8.9 入口50个点
    x_bc_value2 = torch.ones((100, 1)) * domain[2]  # 26.7 #17.8

    y_bc_interval1 = domain[3] + (domain[1] - domain[3]) * torch.rand(50, 1)  # -1.0-0.0 # -1.0 0.0
    y_bc_interval2 = domain[1] + (domain[4] - domain[1]) * torch.rand(50, 1)  # 0.0-1.0 #50个点
    y_bc_interval3 = domain[3] + (domain[4] - domain[3]) * torch.rand(100, 1)  # -1.0-1.0
    y_bc_value1 = torch.ones((500, 1))  # 1.0
    y_bc_value2 = torch.ones((334, 1)) * (-1.0)  # -1.0

    bc_data1 = torch.cat([x_bc_value, y_bc_interval2], 1).to(device)  # x:0 y:[0,1]
    bc_data2 = torch.cat([x_bc_interval2, y_bc_value1], 1).to(device)  # x:[0,26.7] y:1
    bc_data3 = torch.cat([x_bc_interval3, y_bc_value2], 1).to(device)  # x:[8.9,26.7] y:-1
    bc_data4 = torch.cat([x_bc_interval1, bc_zeros], 1).to(device)  # x:[0.0,8.9] y:0
    bc_data5 = torch.cat([bc_zeros_1, y_bc_interval1], 1).to(device)  # x=8.9 y[-1,0]
    bc_data6 = torch.cat([x_bc_value2, y_bc_interval3], 1).to(device) # x=0 y0

    return bc_data1, bc_data2, bc_data3, bc_data4, bc_data5, bc_data6


def equation_data(num_samples, domain):
    SEED = 1
    torch.manual_seed(SEED)
    num =int(num_samples*0.5)

    x_collocation1 = domain[0] + (domain[1] - domain[0]) * torch.rand(1500, 1, requires_grad=True)   # 0.0-8.9
    x_collocation2 = domain[1] + (domain[2] - domain[1]) * torch.rand(3500, 1, requires_grad=True)  # 8.9-26.7

    y_collocation1 = domain[1] + (domain[4] - domain[1]) * torch.rand(1500, 1, requires_grad=True)  # 0.0-1.0
    y_collocation2 = domain[3] + (domain[4] - domain[3]) * torch.rand(3500, 1, requires_grad=True)  # -1.0-1.0

    f_data1 = torch.cat([x_collocation1, y_collocation1], 1).to(device)
    f_data2 = torch.cat([x_collocation2, y_collocation2], 1).to(device)

    return f_data1, f_data2


def supervision_data(path, u1, l1):
    U = u1
    L = l1
    rou = 998.2

    data = pd.read_csv(path)
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

    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)
    u = u.reshape(-1, 1)
    v = v.reshape(-1, 1)
    p = p.reshape(-1, 1)

    x = torch.from_numpy(x)
    x = (x / L).to(torch.float32).to(device)
    x.requires_grad = True
    y = torch.from_numpy(y)
    y = (y / L).to(torch.float32).to(device)
    y.requires_grad = True
    u = torch.from_numpy(u)
    u = (u / U).to(torch.float32).to(device)
    v = torch.from_numpy(v)
    v = (v / U).to(torch.float32).to(device)
    p = torch.from_numpy(p)
    p = (p / (rou * U * U)).to(torch.float32).to(device)

    input_data = torch.cat([x, y], 1)
    real_data = torch.cat([u, v, p], 1)

    return input_data, real_data







