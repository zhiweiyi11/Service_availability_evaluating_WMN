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

from Evaluating_Scripts.Calculating_Availability import *
from Evolution_Model.Evolution_Objects import *
from Evolution_Model.Evolution_Conditions import *
from Evolution_Model.Evolution_Rules import *
from Evaluating_Scripts.Calculating_Availability import *

def save_results(origin_df, file_name):
    # 保存仿真的数据
    # 将dataframe中的数据保存至excel中
    # localtime = time.asctime(time.localtime(time.time()))
    time2 = datetime.datetime.now().strftime('%Y_%m_%d_') # 记录数据存储的时间
    sys_path = os.path.abspath('..')  # 表示当前所处文件夹上一级文件夹的绝对路径
    #
    # with pd.ExcelWriter(r'..\Results_saved\{}_time{}.xlsx'.format(file_name, time2)) as xlsx: # 将紧跟with后面的语句求值给as后面的xlsx变量，当with后面的代码块全部被执行完之后，将调用前面返回对象的exit()方法。
    #     origin_df.to_excel(xlsx, sheet_name='app_avail', index=False) # 不显示行索引
    origin_df.to_excel(r'..\Results_Output\{}_{}.xlsx'.format(file_name, time2), index=False)
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

def calculate_MTBF_analysis(MTTF_list, N, G, App_set, App_priority_list):
    # 业务可用性的MTTF敏感性分析
    MLife = 800
    mttf_single_avail = pd.DataFrame(index=[1,2,3]) # 存储业务可用度数据的行索引为各SLA的值
    mttf_whole_avail = pd.DataFrame(index=['{}次演化'.format(N)]) # 存储业务可用度数据的行索引为演化次数的值

    for mttf in MTTF_list:
        print('当前计算的MTTF值为{} \n'.format(mttf))

        start_time = time.time()
        res = Apps_Availability_MC(N, T,  G, App_set, mttf, MLife, MTTR, detection_rate, message_processing_time,   path_calculating_time, beta, demand_th)
        end_time = time.time()
        print('采用普通蒙卡计算{}次网络演化的时长为{}s \n'.format(N, end_time-start_time))

        SLA_avail = calculate_SLA_results(App_set, res[0], App_priority_list)
        # whole_avail = res[1].apply(np.mean, axis=1) # 对dataframe中的每一行应用求平均值
        whole_avail = np.mean(res[1].iloc[0].tolist())
        mttf_single_avail.loc[:, mttf] = pd.Series(SLA_avail) # 每一列存储该MTTF值下的业务可用度
        mttf_whole_avail.loc[:, mttf] = whole_avail

    return mttf_single_avail, mttf_whole_avail

def draw_line_plot(x_data, y_data, file_name):
    time2 = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M') # 记录数据存储的时间

    x_data = x_data
    y_data = y_data# 不同SLA下的业务可用度

    fig1, ax1 = plt.subplots()
    fig1.subplots_adjust(hspace=0.5) # make a little extra space between the subplots
    # colors = ['gold','blue','green'] # ,'orangered','hotpink'
    if len(y_data.index) > 1: # 如果纵坐标有多行数据
        for i in y_data.index:
            ax1.plot(x_data, y_data.loc[i], label='${}$'.format(i)) # i+1表示业务等级 c=colors[i]
    else:
        ax1.plot(x_data, y2_data)
    ax1.set_xlabel('MTTF of components')
    ax1.set_ylabel('Service Availability')
    # plt.legend(title="Priority")
    plt.savefig(r'..\Pictures_Saved\折线图{}.jpg'.format('{}'.format(file_name), dpi=1200) )
    plt.show()


if __name__ == '__main__':
    # 网络演化对象的输入参数；
    save_data = False # 是否保存节点坐标数据
    Node_num = 200
    Area_size = (250, 250)
    Area_width, Area_length = 250, 250
    Coordinates = generate_positions(Node_num, Area_width, Area_length, save_data)

    # TX_range = 50 # 传输范围为区域面积的1/5时能够保证网络全联通
    transmit_prob = 0.1 # 节点的传输概率
    transmit_power = 1.5  # 发射功率(毫瓦)，统一单位：W
    path_loss = 2  # 单位：无
    noise = pow(10, -11)  # 噪声的功率谱密度(毫瓦/赫兹)，统一单位：W/Hz, 参考自https://dsp.stackexchange.com/questions/13127/snr-calculation-with-noise-spectral-density
    bandwidth = 10 * pow(10, 6)  # 带宽(Mhz)，统一单位：Hz
    lambda_TH = 8 * pow(10, -1)  # 接收器的敏感性阈值,用于确定节点的传输范围
    # TX_range = pow((transmit_power / (bandwidth * noise * lambda_TH)), 1 / path_loss)
    TX_range = 30
    CV_range = 30  # 节点的覆盖范围
    # 网络演化对象的输入参数；
    import_file = False # 不从excel中读取网络拓扑信息
    Topology = 'Random_SINR'


    # 业务请求的参数
    # App_num = 20
    # grid_size = 5
    # traffic_th = 0.5  # 业务网格的流量阈值
    # App_Demand = np.random.normal(loc=3, scale=1, size=App_num)  # 生成平均值为5，标准差为1的业务带宽请求的整体分布
    # App_Priority = [1, 2, 3,4,5]
    # ratio_str = 1  # 尽量分离和尽量重用的业务占比
    # Strategy_P = ['Global'] * int(App_num * (1 - ratio_str))
    # Strategy_S = ['Local'] * int(App_num * ratio_str)
    # App_Strategy = Strategy_S + Strategy_P

    # G = Network(Topology, transmit_prob, Coordinates, TX_range, transmit_power, bandwidth, path_loss, noise, import_file)
    # G, Apps = init_func(G, Coordinates, Area_size, CV_range, grid_size, traffic_th, App_num, App_Demand, App_Priority, App_Strategy)
    # 从文件中创建网络和业务对象
    Network_parameters = [Topology, transmit_prob]
    Wireless_parameters = [TX_range, transmit_power, bandwidth]
    Loss_parameters = [path_loss, noise]

    G, Apps = init_function_from_file('Topology_100_Band=20', 'Node_Coordinates_100_Uniform','App_30_Demand=5', Network_parameters, Wireless_parameters, Loss_parameters)

    # 将业务的策略修改为Global
    # for app_id in range(len(Apps)):
    #     Apps[app_id].str = 'Global'


    # 业务可用性评估的参数
    T = 8760
    MTTF, MLife = 1000, 800
    MTTR = 2
    ## 重路由相关的参数
    message_processing_time = 0.05 # 单位为秒 50ms
    path_calculating_time = 5 # 单位为秒 s
    detection_rate = 0.99
    demand_th = 1*0.2 # 根据App_demand中的均值来确定
    beta = 0.5 # 2类可用性指标的权重(beta越大表明 时间相关的服务可用性水平越重要)

    # 业务可用度评估计算
    N = 50 # 网络演化的次数
    # MTTF, MLife = 1000, 800
    MTBF_list = np.linspace(1000, 2000, 51) # 100个点

    Res = calculate_MTBF_analysis(MTBF_list, N, G, Apps)


    # 对计算结果进行图形化的展示
    time2 = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M') # 记录数据存储的时间

    x_data = MTBF_list
    y_data = Res[0] # 不同SLA下的业务可用度
    y2_data = Res[1]

    fig1, ax1 = plt.subplots()
    fig1.subplots_adjust(hspace=0.5) # make a little extra space between the subplots
    colors = ['gold','blue','green'] # ,'orangered','hotpink'
    for i in range(len(colors)):
        ax1.plot(x_data, y_data.loc[i+1], c=colors[i], label='${}$'.format(i+1)) # i+1表示业务等级
    # ax1.plot(x_data, y2_data.iloc[0], label='whole network')
    ax1.set_xlabel('MTTF of components')
    ax1.set_ylabel('Service Availability')
    plt.legend(loc="upper right", title="Priority")
    plt.savefig(r'..\Pictures_Saved\line_plot_{}.jpg'.format('MTTF敏感性分析[Band=20,Demand=5]-Local演化N={}次-节点数量为{}'.format(N, len(G)), dpi=1200))

    plt.show()

    save_results(Res[0], 'MTTF敏感性分析[Band=20,Demand=5]-Local演化N=10次,100节点的拓扑')



