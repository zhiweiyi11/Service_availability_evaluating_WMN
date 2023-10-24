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
from Evaluating_Scripts.MTBF_sensitivity_analysis import *
from concurrent.futures import ThreadPoolExecutor

def calculating_Coefficient_of_Variation(sample_data):
	# 计算样本的方差系数
	N = len(sample_data)
	var = np.var(sample_data, ddof=1) # ddof=1表示计算样本方差, ddof=0表示总体方差
	cv = np.sqrt(var) / np.mean(sample_data)
	return cv


def convergence_analysis(N, T, G, Apps, App_priority_list):
	# 模型业务可用度计算结果的收敛性分析，对每一个业务都保存其仿真的可用度结果
	sla_avail_df = pd.DataFrame(index=App_priority_list) # 每行为业务的id，每列存储业务可用度的方差系数
	whole_avail_df = pd.DataFrame(index=['整网平均']) # 存储n次演化下各业务带宽损失的均值
	""" ## 多次演化下的业务可用度结果 """
	app_priority = App_priority_list * int(len(Apps) / len(App_priority_list))  # 乘以每类SLA等级下的业务数量
	random.shuffle(app_priority)
	for i in range(len(Apps)):  # 将业务的优先级设置为 [1~5]
		Apps[i].SLA = app_priority[i]
		# Apps[i].str = 'Global'
	for n in range(10, N, 5):
		st2_ = time.time()

		# Availability_Results, Loss_Results = Apps_Availability_MC(n, calculateAvailability, G, App, MTTF, MLife, MTTR, switch_time, switch_rate, survival_time)
		single_results, whole_results = Apps_Availability_MC(n, T, G, Apps, MTTF, MLife, MTTR, detection_rate,message_processing_time, path_calculating_time, beta_list, demand_th)
		single_avail = single_results.apply(calculating_Coefficient_of_Variation, axis=1)  # 对每一行数据进行求变异系数，即得到N次演化下各等级业务的方差系数
		whole_avail = whole_results.apply(calculating_Coefficient_of_Variation, axis=1)

		sla_avail = calculate_SLA_results(Apps, single_avail, App_priority_list) # 对SLA下所有业务的方差系数求平均值作为该SLA等级业务可用度的方差系数

		sla_avail_df.loc[:, n] = pd.Series(sla_avail)
		whole_avail_df.loc[:, n] = whole_avail['evo_times'] # 对每一行数据求均值

		et2_ = time.time()
		print('\n 第{}次演化下并行计算的时长为{}s'.format(n, et2_ - st2_))

	return sla_avail_df, whole_avail_df



def save_results(origin_df, file_name):
	# 保存仿真的数据
	# 将dataframe中的数据保存至excel中
	# localtime = time.asctime(time.localtime(time.time()))
	time2 = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M')  # 记录数据存储的时间
	# sys_path = os.path.abspath('..')  # 表示当前所处文件夹上一级文件夹的绝对路径

	with pd.ExcelWriter( r'..\Results_Output\Convergence_results\{}_time{}.xlsx'.format(file_name, time2)) as xlsx:  # 将紧跟with后面的语句求值给as后面的xlsx变量，当with后面的代码块全部被执行完之后，将调用前面返回对象的exit()方法。
		origin_df.to_excel(xlsx, sheet_name='app_avail', index=False)  # 不显示行索引
		print('数据成功保存')


if __name__ == '__main__':
	# 网络演化对象的输入参数；
	# 网络演化对象的输入参数；
	save_data = False  # 是否保存节点坐标数据
	Node_num = 100
	Area_size = (150, 150)
	Area_width, Area_length = 150, 150
	Coordinates = generate_positions(Node_num, Area_width, Area_length, save_data)

	# TX_range = 50 # 传输范围为区域面积的1/5时能够保证网络全联通
	transmit_prob = 0.1  # 节点的传输概率
	transmit_power = 1.5  # 发射功率(毫瓦)，统一单位：W
	path_loss = 2  # 单位：无
	noise = pow(10, -11)  # 噪声的功率谱密度(毫瓦/赫兹)，统一单位：W/Hz, 参考自https://dsp.stackexchange.com/questions/13127/snr-calculation-with-noise-spectral-density
	bandwidth = 10 * pow(10, 6)  # 带宽(Mhz)，统一单位：Hz
	lambda_TH = 8 * pow(10, -1)  # 接收器的敏感性阈值,用于确定节点的传输范围
	# TX_range = pow((transmit_power / (bandwidth * noise * lambda_TH)), 1 / path_loss)
	TX_range = 30
	CV_range = 30  # 节点的覆盖范围

	import_file = False  # 不从excel中读取网络拓扑信息
	Topology = 'Random_SINR'
	# 从文件中创建网络和业务对象
	Network_parameters = [Topology, transmit_prob]
	Wireless_parameters = [TX_range, transmit_power, bandwidth]
	Loss_parameters = [path_loss, noise]

	# topology_file = 'Topology_100_Band=10[for_priority_analysis]'
	topology_file_mesh = '../Results_Saving/Small_Scale_Network/topo_mesh'

	coordinates_file = 'Node_Coordinates_100_Uniform[for_priority_analysis]'
	# app_file = 'App_50_Demand=2_inTopo=100[for_priority_analysis]'
	app_file_mesh = '../Results_Saving/Small_Scale_Network/App_2_mesh'

	G, Apps = init_function_from_file(topology_file_mesh, coordinates_file, app_file_mesh, Network_parameters,  Wireless_parameters, Loss_parameters)

	# 业务可用性评估的参数
	T = 8760
	MTTF, MLife = 2000, 800
	MTTR = 4
	## 重路由相关的参数
	message_processing_time = 0.05 # 0.05  # 单位为秒s [毫秒量级]
	path_calculating_time = 0.5  # 单位为秒 s [秒量级]
	detection_rate = 0.999
	demand_th = 2*math.pow((1/1),1)*math.exp(-1)  # 根据App_demand中的均值来确定
	beta_list = [0.5]  # 2类可用性指标的权重(beta越大表明 时间相关的服务可用性水平越重要)
	App_priority_list = [1,  5]
	# app_priority = App_priority_list * int(len(Apps) / len(App_priority_list))
	# random.shuffle(app_priority)
	# for i in range(len(Apps)):  # 将业务的优先级设置为 [1~5]
	# 	Apps[i].SLA = app_priority[i]

	# 收敛性分析的参数
	N = 500

	Con_Results = convergence_analysis(N, T, G, Apps, App_priority_list)
	# 结果保存
	save_results(Con_Results[0], '{}次仿真的不同优先级服务可用度方差系数'.format(N))
	save_results(Con_Results[1], '{}次仿真的整网服务可用度的方差系数'.format(N))

	# 绘制收敛性分析的结果图
	font = {'family': 'serif',
			'color': 'darkred',
			'weight': 'normal',
			'size': 16,
			}
	## 服务可用度的方差系数
	x = Con_Results[0].columns # df的列索引，表示仿真次数N
	fig1, ax1 = plt.subplots() # Create a figure and a set of subplots.
	for i in range(5):
		y = Con_Results[0].iloc[i]
		ax1.plot(x, y, label='$s_{}$'.format(i+1))
	# plt.title('Loss of Service Exception (LOSE)', fontdict=font)
	plt.xlabel('simulation times ($N$)', fontdict=font)
	plt.ylabel('Coefficient of variation', fontdict=font)
	plt.legend(loc='upper right',title='priority')
	plt.ylim(top=0.01)
	# Tweak spacing to prevent clipping of ylabel 调整间距以防止剪裁ylabel
	plt.subplots_adjust(left=0.15)
	plt.savefig(r'..\Pictures_Saved\收敛性分析图{}.jpg'.format('{}'.format('不同SLA的服务A方差系数'), dpi=1200))

	plt.show()

	## 服务带宽损失

	fig, ax = plt.subplots() # Create a figure and a set of subplots.
	y = Con_Results[1].loc['整网平均']
	ax.plot(x, y, label='$s_{}$'.format(i))
	# plt.title('Loss of Service Exception (LOSE)', fontdict=font)
	plt.xlabel('simulation times ($N$)', fontdict=font)
	plt.ylabel('coefficient of variation', fontdict=font)
	plt.subplots_adjust(left=0.15)
	plt.ylim(top=0.01)
	plt.legend(title='whole network')
	plt.savefig(r'..\Pictures_Saved\收敛性分析图{}.jpg'.format('{}'.format('整网服务A方差系数'), dpi=1200))
	plt.show()


