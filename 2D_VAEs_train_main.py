# 这是一个开头
# 人员：sulei
# 开发时间：18/11/2020下午8:09
# 文件名：2D_VAEs_train_main.py
# 开发工具：PyCharm

from collections import OrderedDict
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
import time

from mydocument.hw3_helper import *
from mydocument.VAE_funtions import *

'''Part (a) Data from a Full Covariance Gaussian '''
#通过数据训练VAE模型，数据是由高斯模型产生的，并且高斯混合模型拥有协方差矩阵？？
#协方差矩阵： 比如有x,y，z三个变量，那么他们的协方差矩阵就是一个3×3的，并且对角线为cov(x,x),cov(y,y),cov(z,z)
#           位置（1，2） 为 cov (x,y)  位置(2,1) 为 cov(y,x)等等

# 功能1：可视化产生的数据
visualize_q1_data('a', 1)
visualize_q1_data('a', 2)



#功能2：训练一个VAE模型

t_start = time.time()

print(t_start)
q1_save_results('a', 1, q1)

q1_save_results('a', 2, q1)

t_end = time.time()
print(t_end)

print(t_end - t_start)