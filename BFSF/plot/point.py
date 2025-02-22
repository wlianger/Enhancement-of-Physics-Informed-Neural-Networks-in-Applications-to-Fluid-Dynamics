from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2

current_time = datetime.now()
filename = current_time.strftime("%Y-%m-%d")
result_path = "../result/"+filename+"/"

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['font.weight'] = 'bold'

data = pd.read_csv("../data/dataset/U=0.02-decrease/BF_train_0.0050.csv")
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
plt.scatter(x/0.01, y/0.01, s=10)

# data = pd.read_csv("data/Engineering-data/U=0.05-new/12-10 - 02.csv")
#
# col_1 = data["x"]
# x = np.array(col_1)
# col_2 = data["y"]
# y = np.array(col_2)
# col_3 = data["u"]
# u = np.array(col_3)
# col_4 = data["v"]
# v = np.array(col_4)
# col_5 = data["p"]
# p = np.array(col_5)
#
# plt.scatter(x, y)
#
# data = pd.read_csv("data/Engineering-data/U=0.05-new/12-10 - 03.csv")
#
# col_1 = data["x"]
# x = np.array(col_1)
# col_2 = data["y"]
# y = np.array(col_2)
# col_3 = data["u"]
# u = np.array(col_3)
# col_4 = data["v"]
# v = np.array(col_4)
# col_5 = data["p"]
# p = np.array(col_5)
#
# plt.scatter(x, y)


plt.xlim(-0.089/0.01, 0.178/0.01)
plt.ylim(-0.01/0.01, 0.01/0.01)
# plt.title("h/H=10  Distribution of sample points", size=15)
plt.xlabel("x", size=20, fontstyle='italic')
plt.ylabel("y", size=20, fontstyle='italic')

plt.tick_params(labelsize=15)

# plt.savefig(result_path + "points_0.0090" + ".png", bbox_inches='tight', dpi=300, pad_inches=0.1)
plt.show()


