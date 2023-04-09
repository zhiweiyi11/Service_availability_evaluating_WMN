#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   ：Service_availability -> MTBF_sensitivity_analysis
@IDE    ：PyCharm
@Author ：Yi Zhiwei
@Date   ：2023/4/5 19:28
@Desc   ：不同构件MTTF值下业务可用度结果的敏感性分析
=================================================='''

import numpy as np
import networkx as nx
import random
import pandas as pd
from Evolution_Model.Evolution_Objects import *
from Evolution_Model.Evolution_Conditions import *
from Evolution_Model.Evolution_Rules import *
from Evaluating_Scripts.Calculating_Availability import *

def save_results(origin_df, file_name):
    # 保存仿真的数据
    # 将dataframe中的数据保存至excel中
    # localtime = time.asctime(time.localtime(time.time()))
    time2 = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M') # 记录数据存储的时间
    sys_path = os.path.abspath('..')  # 表示当前所处文件夹上一级文件夹的绝对路径
    #
    # with pd.ExcelWriter(r'..\Results_saved\{}_time{}.xlsx'.format(file_name, time2)) as xlsx: # 将紧跟with后面的语句求值给as后面的xlsx变量，当with后面的代码块全部被执行完之后，将调用前面返回对象的exit()方法。
    #     origin_df.to_excel(xlsx, sheet_name='app_avail', index=False) # 不显示行索引
    origin_df.to_excel(r'..\Results_Saving\{}_{}.xlsx'.format(file_name, time2), index=False)
    print('数据成功保存')

def calculate_SLA_results(multi_res):
    # 计算不同SLA等级下的业务可用度\业务带宽损失
    SLA_res = {}
    for i in range(1, 6):
        res = multi_res.iloc[(i-1)*20:i*20].apply(np.mean, axis =1 )# 对N次演化下业务可用度的结果求均值，对dataFrame的每行进行操作
        SLA_res[i] = np.mean(res) # 对SLA等级下的所有业务可用度的均值再求均值，作为该等级下的业务可用度
    return SLA_res

def calculate_MTBF_analysis(MTTF_list, N, G, App_set):
    # 业务可用性的MTTF敏感性分析
    MLife = 800
    multi_SLA_avail = pd.DataFrame(index=[1,2,3,4,5]) # 存储业务可用度数据的行索引为各SLA的值
    multi_SLA_loss = pd.DataFrame(index=[1,2,3,4,5]) # 存储业务可用度数据的行索引为各SLA的值

    for mttf in MTTF_list:
        print('当前计算的MTTF值为{} \n'.format(mttf))

        start_time = time.time()
        res = Apps_Availability_MC(N, T,  G, App_set, mttf, MLife, MTTR, switch_time, switch_rate, survival_time)
        end_time = time.time()
        print('采用普通蒙卡计算{}次网络演化的时长为{}s \n'.format(N, end_time-start_time))

        SLA_avail = calculate_SLA_results(res[0])
        SLA_loss = calculate_SLA_results(res[1])
        multi_SLA_avail.loc[:, mttf] = pd.Series(SLA_avail) # 每一列存储该MTTF值下的业务可用度
        multi_SLA_loss.loc[:, mttf] = pd.Series(SLA_loss)

    return multi_SLA_avail, multi_SLA_loss



if __name__ == '__main__':
    ## 网络层对象
    Topology = 'Random'
    Node_num, App_num = 100, 50
    Capacity = 50
    Demand = np.random.normal(loc=10, scale=2, size=App_num)  # 生成平均值为5，标准差为1的带宽的正态分布
    Area_width, Area_length = 250, 150
    Area_size = (250, 150)

    TX_range = 50  # 传输范围为区域面积的1/5时能够保证网络全联通
    CV_range = 30
    Coordinates = generate_positions(Node_num, Area_width, Area_length)
    # Demand = list(map(int, Demand)) # 将业务的带宽需求换成整数
    ## 业务层对象
    grid_size = 5
    traffic_th = 0.5
    Priority = np.linspace(start=1, stop=5, num=5, dtype=int)
    ratio_str = 0.5  # 尽量分离和尽量重用的业务占比
    Strategy_P = ['Global'] * int(App_num * (1 - ratio_str))
    Strategy_S = ['Local'] * int(App_num * ratio_str)
    Strategy = Strategy_S + Strategy_P

    # 演化条件的参数
    T = 8760
    # MTTF, MLife = 1000, 800
    MTBF_list = np.linspace(1000, 2000, 100)

    MTTR = 2

    ## 重路由相关的参数
    switch_time = 10
    switch_rate = 0.99
    survival_time = 3 * switch_time  # 允许的最大重路由次数为3次

    # 初始化网络演化对象
    G, App = init_func(True, 'App_Info_Local', Area_size, Node_num, Topology, TX_range, CV_range, Coordinates, Capacity, grid_size, App_num, traffic_th, Demand, Priority, Strategy)
    # 生成网络演化条件
    N = 200

    Res = calculate_MTBF_analysis(MTBF_list, N, G, App)
