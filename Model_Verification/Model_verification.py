#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   ：Service_availability -> Model_verification
@IDE    ：PyCharm
@Author ：Yi Zhiwei
@Date   ：2023/7/18 16:40
@Desc   ：模型验证：对比考虑恢复与不考虑恢复下的服务可用性结果
=================================================='''


import numpy as np
import networkx as nx
import random
import pandas as pd

from Evaluating_Scripts.Calculating_Availability import *
from Evolution_Model.Evolution_Objects import *
from Evolution_Model.Evolution_Conditions import *
from Evolution_Model.Evolution_Rules import *
from Evaluating_Scripts.Calculating_Availability import *

def save_results(origin_df, file_name):
	# 保存仿真的数据
	# 将dataframe中的数据保存至excel中
	# localtime = time.asctime(time.localtime(time.time()))
	time2 = datetime.datetime.now().strftime('%Y_%m_%d_+%H_%M') # 记录数据存储的时间
	sys_path = os.path.abspath('..')  # 表示当前所处文件夹上一级文件夹的绝对路径
	#
	# with pd.ExcelWriter(r'..\Results_saved\{}_time{}.xlsx'.format(file_name, time2)) as xlsx: # 将紧跟with后面的语句求值给as后面的xlsx变量，当with后面的代码块全部被执行完之后，将调用前面返回对象的exit()方法。
	#     origin_df.to_excel(xlsx, sheet_name='app_avail', index=False) # 不显示行索引
	origin_df.to_excel(r'..\Results_Output\Model_verification\{}_{}.xlsx'.format(file_name, time2), index=False)
	print('数据成功保存')

def calculate_SLA_results(Apps, multi_app_res, app_priority_list):
	# 计算不同SLA等级下的业务可用度\业务带宽损失
	App_id_SLA = {}  # 存储各优先级下的业务可用度结果
	for pri in app_priority_list:
		App_id_SLA[pri] = []  # 为每个优先级下的业务创建一个空列表来存储统计的各SLA下业务的可用度
	## 1.先统计不同SLA对应的业务id
	for i in range(len(Apps)):
		SLA = Apps[i].SLA
		App_id_SLA[SLA].append(i)

	## 2.再计算不同SLA下各业务的平均值( average=(该SLA下单个业务的N次演化平均)/Num_SLA )
	SLA_res = {}
	for SLA, app_list in App_id_SLA.items():
		Ave_SLA = [] # 统计各SLA下业务多次演化的平均值
		for app_id in app_list:
			res = np.mean(multi_app_res.loc[app_id])
			Ave_SLA.append(res)
		SLA_res[SLA] = np.mean(Ave_SLA) # 对SLA等级下的所有业务可用度的均值再求均值，作为该等级下的业务可用度
	return SLA_res

def calculate_MTTF_comparison_results(MTTF_list, N, G, Apps,  Restoration_metric_list):
	# 计算各MTTF值下网络的服务可用度【考虑恢复与不考虑恢复】

	message_processing_time = 0
	availability_different_restoration_local = pd.DataFrame(index=Restoration_metric_list)  # 存储结果
	availability_different_restoration_global = pd.DataFrame(index=Restoration_metric_list)  # 存储结果

	for mttf in MTTF_list:
		print('当前计算的MTTF值为{} \n'.format(mttf))
		start_time = time.time()
		res_local = []
		for restoration_metric in Restoration_metric_list:
			detection_rate = restoration_metric[0]
			path_calculating_time = restoration_metric[1]
			each_avail_local, whole_avail_local = Apps_Availability_MC(N, T, G, Apps, mttf, MLife, MTTR, detection_rate, message_processing_time,
													  path_calculating_time, beta_list, demand_th)
			res_local.append(np.mean(whole_avail_local.iloc[0].tolist()))
		availability_different_restoration_local.loc[:, mttf] = res_local
		end_time = time.time()
		print('采用普通蒙卡计算{}次网络演化的时长为{}s \n'.format(N, end_time - start_time))

	save_results(availability_different_restoration_local, '模型结果对比-MTTF参数下不同恢复性能指标的服务可用度-{}策略,演化N={}次,{}节点的拓扑'.format(Apps[0].str, N, len(G)))

	for i in range(len(Apps)):  # 将业务的优先级设置为 [1~5]
		Apps[i].str = 'Global'  # 将业务的策略设置为Global

	for mttf in MTTF_list:
		print('当前计算的MTTF值为{} \n'.format(mttf))
		start_time = time.time()
		res_global = []
		for restoration_metric in Restoration_metric_list:
			detection_rate = restoration_metric[0]
			path_calculating_time = restoration_metric[1]
			each_avail_global, whole_avail_global = Apps_Availability_MC(N, T, G, Apps, mttf, MLife, MTTR, detection_rate, message_processing_time,
													  path_calculating_time, beta_list, demand_th)
			res_global.append(np.mean(whole_avail_global.iloc[0].tolist()))

		availability_different_restoration_global.loc[:, mttf] = res_global
		end_time = time.time()
		print('采用普通蒙卡计算{}次网络演化的时长为{}s \n'.format(N, end_time - start_time))

	save_results(availability_different_restoration_global, '模型结果对比-MTTF参数下不同恢复性能指标的服务可用度-{}策略,演化N={}次,{}节点的拓扑'.format(Apps[0].str, N, len(G)))

	return availability_different_restoration_local, availability_different_restoration_global


def calculate_MTTR_comparison_results(MTTR_list, N, G, Apps,  Restoration_metric_list):
	pass



if __name__ == '__main__':

	Restoration_metric_list = [(1, 0), (0.9, 5), (0.99, 5),  (0.999, 5), (0.9, 30), (0.99, 30), (0.999, 30)]
	## 无线传输相关的参数
	transmit_prob = 0.1  # 节点的传输概率
	transmit_power = 1.5  # 发射功率(毫瓦)，统一单位：W
	path_loss = 2  # 单位：无
	noise = pow(10,
				-11)  # 噪声的功率谱密度(毫瓦/赫兹)，统一单位：W/Hz, 参考自https://dsp.stackexchange.com/questions/13127/snr-calculation-with-noise-spectral-density
	bandwidth = 10 * pow(10, 6)  # 带宽(Mhz)，统一单位：Hz
	lambda_TH = 8 * pow(10, -1)  # 接收器的敏感性阈值,用于确定节点的传输范围
	TX_range = 30
	CV_range = 30  # 节点的覆盖范围
	Topology = 'Random_SINR'

	## 1. 业务的优先级分析
	App_priority_list = [1, 2, 3, 4, 5]
	topology_file = 'Topology_100_Band=10[for_priority_analysis]'
	coordinates_file = 'Node_Coordinates_100_Uniform[for_priority_analysis]'
	app_file = 'App_50_Demand=2_inTopo=100[for_priority_analysis]'  # 'App_50_Demand=2_inTopo=100_Band=10' #

	Network_parameters = [Topology, transmit_prob]
	Wireless_parameters = [TX_range, transmit_power, bandwidth]
	Loss_parameters = [path_loss, noise]

	## 服务可用性评估相关的参数
	N = 5
	# T = 30 * 24 # 一个月
	T = 876  # 一年时长
	message_processing_time = 0.05  # 单位为秒 50ms
	path_calculating_time = 5  # 单位为秒 s
	detection_rate = 0.99
	demand_th = 0.2  # 根据App_demand中的均值来确定
	beta_list = [0.5]  # 2类可用性指标的权重(beta越大表明 时间相关的服务可用性水平越重要)

	# MTTF = 2000
	MTTR = 4
	MLife = 800
	MTTF_list = np.linspace(1000, 2000, 11)  # 40个点

	G, Apps = init_function_from_file(topology_file, coordinates_file, app_file, Network_parameters, Wireless_parameters,
									  Loss_parameters)

	local_res , global_res = calculate_MTTF_comparison_results(MTTF_list, N, G, Apps, Restoration_metric_list)


# 1. 生成一个mesh拓扑


	# 2. 采用CTMC来建模服务可用度



	# 3. 对比建模的准确性【不考虑恢复机制】



	# 4. 对比考虑恢复性能参数的效果【恢复时间、恢复成功率】
