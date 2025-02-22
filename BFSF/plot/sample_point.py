from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2
from sympy.physics.control.control_plots import matplotlib

current_time = datetime.now()
filename = current_time.strftime("%Y-%m-%d")
result_path = "F:\\project\\BF\\result\\"+filename+"\\"

# 统一设置字体
plt.rcParams["font.family"] = 'Times New Roman'

# 分别设置mathtext公式的正体和斜体字体
matplotlib.rcParams['mathtext.fontset'] = 'custom'
matplotlib.rcParams['mathtext.rm'] = 'Times New Roman'  # 用于正常数学文本
matplotlib.rcParams['mathtext.it'] = 'Times New Roman:italic'  # 用于斜体数学文本

data = pd.read_csv("F:/project/BF/data/dataset/U=0.05-new/1.csv")
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

plt.figure(1, figsize=(9, 3), dpi=300)
plt.scatter(x, y, s=2)

data = pd.read_csv("F:/project/BF/data/dataset/U=0.05-new/2.csv")
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

plt.figure(1, figsize=(9, 3), dpi=300)
plt.scatter(x, y, s=2)

data = pd.read_csv("F:/project/BF/data/dataset/U=0.05-new/3.csv")
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

plt.figure(1, figsize=(9, 3), dpi=300)
plt.scatter(x, y, s=2)
# 
data = pd.read_csv("F:/project/BF/data/dataset/U=0.05-new/4.csv")
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

plt.figure(1, figsize=(9, 3), dpi=300)
plt.scatter(x, y, s=2, color='y')

plt.xlim(0.0, 0.267)
plt.ylim(-0.01, 0.01)
plt.title("d/D=5  Distribution of sample points", size=20)

plt.xlabel(r"$\it{x}$$\rm(m)$", fontsize=20)
plt.ylabel(r"$\it{y}$$\rm(m)$", fontsize=20)

plt.tick_params(labelsize=20)
plt.savefig(result_path + "points_0.0095" + ".png", bbox_inches='tight', dpi=600, pad_inches=0.1)
plt.show()


