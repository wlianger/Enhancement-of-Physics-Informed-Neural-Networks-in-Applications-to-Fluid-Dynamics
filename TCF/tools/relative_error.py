import numpy as np
import torch


def safe_relative_error(true_values, estimated_values, epsilon=1e-10):
    retrue_values = np.where(np.abs(true_values) < epsilon, epsilon, true_values)
    relative_error = np.abs((true_values - estimated_values) / retrue_values)
    clipped_array = np.clip(relative_error, a_min=0, a_max=1)

    return clipped_array


def symmetric_relative_error(actual, predicted, epsilon=1e-6):

    if actual.shape != predicted.shape:
        raise ValueError("实际值和预测值的tensor形状必须相同")

    # 计算对称相对误差
    sre = np.abs(actual - predicted) / (np.abs(actual) + np.abs(predicted) + epsilon)

    # 避免当实际值和预测值都为0时的除以零情况
    sre[(actual == 0) & (predicted == 0)] = 0.0

    return sre

def custom_error(actual, predicted):

    if actual.shape != predicted.shape:
        raise ValueError("实际值和预测值的tensor形状必须相同")

    result = (np.abs(predicted-actual))/(np.sum(np.abs(actual))/len(actual))
    #
    result1 =np.max(np.abs(predicted-actual))
    av = np.sum(np.abs(actual))/len(actual)
    print("Maximum absolute value error:"+str(result1))
    print("The average of the absolute values:"+str(av))
    return result

def REL2(actual, predicted):
    if actual.shape != predicted.shape:
        raise ValueError("实际值和预测值的tensor形状必须相同")
    l2_distance = np.linalg.norm(actual - predicted)
    l2_distance_1 = np.linalg.norm(actual)
    res = l2_distance / l2_distance_1
    return res

def error_one(actual, predicted):
    if actual.shape != predicted.shape:
        raise ValueError("实际值和预测值的tensor形状必须相同")
    difference = predicted - actual
    squared_difference = difference**2
    column = np.sum(squared_difference, axis=1)
    column = column.reshape(-1, 1)
    molecular = np.sqrt(column)
    print(molecular)

    r1 = actual**2
    r11 = np.sum(r1, axis=1)
    r11 = r11.reshape(-1, 1)
    res_r1 = np.sqrt(r11)
    denominator = np.sum(res_r1)/(len(res_r1)*len(res_r1[0]))
    print(denominator)
    res = molecular/denominator
    return res