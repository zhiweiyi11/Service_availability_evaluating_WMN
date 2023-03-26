#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   ：Service_availability -> Convergence_Analysis
@IDE    ：PyCharm
@Author ：Yi Zhiwei
@Date   ：2023/3/24 10:33
@Desc   ：模型收敛性分析的代码（对每一个业务/每一类业务）
=================================================='''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy
import time
import os
import  seaborn as sns

from scipy import stats

from Evolution_Model.Evolution_Objects import *
from Evolution_Model.Evolution_Conditions import *
from Evolution_Model.Evolution_Rules import *
from Evolution_Model.Application_request_generating import *
from Evaluating_Scripts.Calculating_Availability import *

def calculating_Coefficient_of_Variation(sample_data):
	# 计算样本的方差系数
	N = len(sample_data)
	var = np.var(sample_data, ddof=1) # ddof=1表示计算样本方差, ddof=0表示总体方差
	cv = np.sqrt(var) / np.mean(sample_data)
	return cv


def convergence_analysis(N, T, G, App,  MTTF, MLife, MTTR, switch_time, switch_rate, survival_time):
	# 模型业务可用度计算结果的收敛性分析，对每一个业务都保存其仿真的可用度结果
	app_cv_df = pd.DataFrame(index=App.keys()) # 每行为业务的id，每列存储业务可用度的方差系数
	app_loss_df = pd.DataFrame(index=App.keys()) # 存储n次演化下各业务带宽损失的均值
	""" ## 多次演化下的业务可用度结果 """
	for n in range(10, N):
		st2_ = time.time()
		pool_num = 6
		args = [T, G, App, MTTF, MLife, MTTR, switch_time, switch_rate, survival_time]
		Availability_Results, Loss_Results = Apps_availability_func(n, args, pool_num)
		app_avail = Availability_Results.apply(calculating_Coefficient_of_Variation, axis=1)  # 对每一行数据进行求变异系数，即得到N次演化下各等级业务的方差系数
		app_cv_df.loc[:, n] = app_avail
		app_loss_df.loc[:, n] = Loss_Results.mean(axis=1) # 对每一行数据求均值

		et2_ = time.time()
		print('\n 第{}次演化下并行计算的时长为{}s'.format(n, et2_ - st2_))

	return app_cv_df, app_loss_df

def save_results(origin_df, file_name):
	# 保存仿真的数据
	# 将dataframe中的数据保存至excel中
	# localtime = time.asctime(time.localtime(time.time()))
	time2 = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M')  # 记录数据存储的时间
	sys_path = os.path.abspath('..')  # 表示当前所处文件夹上一级文件夹的绝对路径

	with pd.ExcelWriter(sys_path + r'.\Results_saved\{}_time{}.xlsx'.format(file_name, time2)) as xlsx:  # 将紧跟with后面的语句求值给as后面的xlsx变量，当with后面的代码块全部被执行完之后，将调用前面返回对象的exit()方法。
		origin_df.to_excel(xlsx, sheet_name='app_avail', index=False)  # 不显示行索引
		print('数据成功保存')


if __name__ == '__main__':
	# 网络演化对象的输入参数；
	## 网络层对象
	Topology = 'Random'
	Node_num, App_num = 100, 50
	Capacity = 50
	Area_width, Area_length = 250, 150
	Area_size = (250, 150)

	TX_range = 50  # 传输范围为区域面积的1/5时能够保证网络全联通
	CV_range = 30
	Coordinates = generate_positions(Node_num, Area_width, Area_length)
	# Demand = list(map(int, Demand)) # 将业务的带宽需求换成整数

	## 业务层对象
	grid_size = 5
	traffic_th = 0.5
	Demand = np.random.normal(loc=10, scale=2, size=App_num)  # 生成平均值为5，标准差为1的带宽的正态分布
	Priority = np.linspace(start=1, stop=5, num=5, dtype=int)
	ratio_str = 1  # 尽量分离和尽量重用的业务占比
	Strategy_P = ['Repetition'] * int(App_num * (1 - ratio_str))
	Strategy_S = ['Separate'] * int(App_num * ratio_str)
	Strategy = Strategy_S + Strategy_P

	# 演化条件的参数
	T = 8760
	MTTF, MLife = 3000, 2000
	MTTR = 2

	## 重路由相关的参数
	switch_time = 10 # 单位为秒
	switch_rate = 0.99
	survival_time = 3 * switch_time  # 允许的最大重路由次数为5次

	# 初始化网络演化对象
	start_time = time.time()
	G, App = init_func(Area_size, Node_num, Topology, TX_range, CV_range, Coordinates, Capacity, grid_size, App_num, traffic_th, Demand, Priority, Strategy)

	# 收敛性分析的参数
	N = 50

	Con_Results = convergence_analysis(N, T, G, App, MTTF, MLife, MTTR, switch_time, switch_rate, survival_time)


