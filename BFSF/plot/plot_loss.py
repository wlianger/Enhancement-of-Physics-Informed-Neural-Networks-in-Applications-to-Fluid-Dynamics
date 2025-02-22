import os
from datetime import datetime

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from tools.create_folder import create_folder_if_not_exists


def read_loss(path):
    data = pd.read_csv(path)
    col_1 = data["iteration"]
    i = np.array(col_1)
    col_2 = data["total"]
    t = np.array(col_2)
    col_3 = data["bc"]
    f = np.array(col_3)
    col_4 = data["data"]
    b = np.array(col_4)
    col_4 = data["f"]
    d = np.array(col_4)
    col_5 = data["error_l2"]
    e = np.array(col_5)
    return i, t, f, b, d, e


current_time = datetime.now()
filename = current_time.strftime("%Y-%m-%d")
path = "../result/"+filename+"/-0.0099/"
create_folder_if_not_exists(path)

fig = plt.figure(1, figsize=(20, 12))
fig.suptitle("train-tanh-loss", size=20)
fig.subplots_adjust(hspace=0.55)

loss_iteration, loss_total, loss_bc, loss_data, loss_f, loss_val = read_loss('../data/loss/U=0.05/ninit-2/2024-09-01-loss-0.0096.csv')

# loss_iteration1, loss_total1, loss_bc1, loss_data1, loss_f1, loss_val1 = read_loss('../data/loss/U=0.05/original/2024-09-01-loss-0.0090.csv')
af1 = fig.add_subplot(211)
# plt.ylim(0.0, 0.02)
# loss_bc = np.clip(loss_bc, a_min=None, a_max=0.02)
# loss_data = np.clip(loss_data, a_min=None, a_max=0.02)
# loss_f = np.clip(loss_f, a_min=None, a_max=0.02)

plt.plot(loss_iteration, loss_bc, label="loss_bc")
plt.plot(loss_iteration, loss_data, label="loss_data")
plt.plot(loss_iteration, loss_f, label="loss_f")
plt.legend(prop={'size': 15})
plt.xlabel('epoch', size=15)
plt.ylabel('loss', size=15)
plt.yscale('log')
af1.tick_params(axis='y', labelsize=15)
af1.tick_params(axis='x', labelsize=15)

# af2 = fig.add_subplot(212)
# plt.plot(loss_iteration, loss_total, label="PINN-GradNorm")
# plt.plot(loss_iteration1, loss_total1, label="PINN")
# plt.xlabel('epoch', size=15)
# plt.ylabel('loss', size=15)
# af2.tick_params(axis='y', labelsize=15)
# af2.tick_params(axis='x', labelsize=15)
# plt.legend(prop={'size': 15})
# plt.yscale('log')
# plt.savefig(path+'train-tanh-loss')
plt.show()

fig = plt.figure(2, figsize=(20, 12))
fig.suptitle("verify-tanh-loss", size=20)

af1 = fig.add_subplot(111)
plt.plot(loss_iteration, loss_val, label="loss_val")
plt.xlabel('epoch', size=15)
plt.ylabel('loss', size=15)
af1.tick_params(axis='y', labelsize=15)
af1.tick_params(axis='x', labelsize=15)
plt.legend(prop={'size': 15})
plt.show()