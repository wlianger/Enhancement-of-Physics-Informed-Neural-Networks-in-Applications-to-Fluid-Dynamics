import csv
from datetime import datetime

import numpy
from sympy.codegen.ast import float64
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
path_model = "../../model/u=0.5/0.0510/2024-12-06-0.0510-grad_model"
read_base_path = "../../data/dataset/u=0.5/"

U = 0.025
L = 0.05

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

data2 = pd.read_csv(read_base_path+"your_file_updated.csv")

# col_12 = data2["x"]
# x2 = np.array(col_12)
# col_22 = data2["y"]
# y2 = np.array(col_22)
# col_32 = data2["sv"]
# d2 = np.array(col_32)
#
# pt_x2 = Variable(torch.from_numpy(x2).float(), requires_grad=True).unsqueeze(-1)
# pt_y2 = Variable(torch.from_numpy(y2).float(), requires_grad=True).unsqueeze(-1)
# net = PINN_net(2, 256, 4, 7)
# net.load_state_dict(torch.load(path_model))
# result2 = net(torch.cat([pt_x2/L, pt_y2/L], 1))
# re_s = result2[:, 1].unsqueeze(-1)*U
# res = re_s.detach().numpy()
#
# data2['pre_us'] = res
# data2.to_csv('your_file_updated_1.csv', index=False)

group_df = data2.groupby('y').mean()
print("\n按第二列分组并求平均值后的数据:")
print(group_df)

output_file_path = read_base_path+'grouped_data.csv'
group_df.to_csv(output_file_path)
print(f"\n处理后的数据已保存到 {output_file_path}")


