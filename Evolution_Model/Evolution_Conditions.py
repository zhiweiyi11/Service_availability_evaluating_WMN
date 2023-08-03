#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   ：Service_availability -> Evolution_Conditions
@IDE    ：PyCharm
@Author ：Yi Zhiwei
@Date   ：2023/3/6 16:50
@Desc   ： 生成网络演化条件
=================================================='''
#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   ：Minimum_Cut -> component_state_sampling
@IDE    ：PyCharm
@Author ：Yi Zhiwei
@Date   ：2023/2/24 17:22
@Desc   ： 在任意分布下，对构件故障与修复状态的采样
=================================================='''

import numpy as np
import matplotlib.pyplot as plt
import time
import random
import pandas as pd
from Evolution_Model.Evolution_Objects import *


def monte_carlo_sequential(nodes_info, sample=False, time_interval=12, T = 365 * 24):
    # 序贯状态连续抽样法，根据链路的MTBF和MTTR数据获取其演化态
    def fail_state(x):
        # lam = (x['MTTF']*x['MLife']) / (x['MTTF'] + x['MLife'])  # 这里的故障率的计算是否应该用MTTF
        lam = 1/ x['MTTF']  # + 1/x['MLife'] # 暂不考虑节点的电池寿命
        miu = 1 / x['MTTR']
        t = 0
        fail_time = []
        recover_time = []
        while True:
            if t >= T:
                break
            else:
                delta = random.random()
                t_f = -np.log(delta) / lam
                # print('fail_rate_time {}'.format(t_f))

                t_r = -np.log(delta) / miu  # 修复时长为2h，不同的采样需要设置不同的随机数
                # print('构件修复时长为{}'.format(t_r))
                # t_r = x['MTTR(小时)']  # 先按照固定的维修时间
                if (t + t_f) <= T:
                    fail_time.append(t + t_f)
                else:
                    break
                if (t + t_f + t_r) <= T:
                    recover_time.append(t + t_f + t_r)
                else:
                    break
                t = t + t_f + t_r
        return fail_time, recover_time # 返回链路在演化时间T内的“故障”与“修复”时刻

    temp = nodes_info.apply(fail_state, axis=1)
    # print('每条链路故障恢复状态已确定')
    nodes_info['fail_begin'] = temp.apply(lambda x: x[0])
    nodes_info['repair_begin'] = temp.apply(lambda x: x[1])
    nodes_info1 = nodes_info.copy()  # 用来表示按照换算的时间间隔采样的数据

    if sample == True:
        # time_interval = 12 # 一小时内有多少个时间步，此时所有时间换算成5分钟
        nodes_info1['fail_begin'] = nodes_info['fail_begin'].apply(lambda x: [int(i * time_interval) for i in x])

        nodes_info1['repair_begin'] = nodes_info['repair_begin'].apply(lambda x: [int(i * time_interval) for i in x])

    # evol = pd.DataFrame(columns=['故障','修复'])

    time_set = set([])
    for i in nodes_info1['fail_begin']: time_set = time_set | set(i)
    for i in nodes_info1['repair_begin']: time_set = time_set | set(i)
    time_set = [i for i in time_set]
    time_set.sort()
    # evol = pd.DataFrame(index=time_set,columns=['故障','修复']).sort_index()

    # print('演化态共有个%d' % len(time_set))

    t = 0
    a = list(nodes_info1['fail_begin'])
    b = list(nodes_info1['repair_begin'])
    all_fail_node_set = []
    all_recover_node_set = []
    if time_set == []:
        all_fail_node_set = []
        all_recover_node_set = []
    else:
        while t < time_set[-1]:
            fail_time_list = [i[0] if i != [] else T for i in a]
            # 记录当前节点/边状态改变(故障)的最先时间
            recover_time_list = [i[0] if i != [] else T for i in b]
            # 记录当前节点/边状态改变(修复)的最先时间
            fail_time = min(fail_time_list)
            recover_time = min(recover_time_list)
            fail_node_index = fail_time_list.index(fail_time)
            recover_node_index = recover_time_list.index(recover_time)
            fail_set = []
            recover_set = []
            if fail_time < recover_time:
                fail_set.append(nodes_info1.loc[fail_node_index, 'Node_id'])
                t = fail_time
                a[fail_node_index] = a[fail_node_index][1:]

            elif fail_time > recover_time:
                recover_set.append(nodes_info1.loc[recover_node_index, 'Node_id'])
                t = recover_time
                b[recover_node_index] = b[recover_node_index][1:]

            all_fail_node_set.append(fail_set)
            all_recover_node_set.append(recover_set)

        # print('正在写入文件')
    evol_con = pd.DataFrame([all_fail_node_set, all_recover_node_set])
    evol_con = evol_con.T
    evol_con.index = time_set
    evol_con.columns = ['fail', 'repair']

    return evol_con # 返回网络的演化条件

def cond_func(G, MTTF, MLife, MTTR, T):
    # 生成网络的演化条件, T为网络演化的总时长
    Num_nodes = len(G.nodes()) # 排除sink节点
    MTTF_list = np.full(Num_nodes, MTTF)
    MLife_list  = np.full(Num_nodes, MLife)
    # MTTF_list = np.random.normal(loc=MTTF, scale=MTTF/4, size=Num_edges)
    # 确保每个值为正数
    for i in range(Num_nodes):
        if MTTF_list[i] < 0:
            MTTF_list[i] = MTTF
    MTTR_list = np.full(Num_nodes, MTTR)

    # T = 365 * 24  # 网络演化的时长
    # Nodes = list(range(Num_nodes))  # 网络节点集的列表
    Nodes = list(G.nodes())
    # Nodes_hash = {i: Nodes[i] for i in range(Num_nodes)}
    nodes_info = pd.DataFrame({
        "Node_id": Nodes,
        "MTTF": MTTF_list,
        "MLife": MLife_list,
        "MTTR": MTTR_list,
        'fail_begin': list(np.full(Num_nodes, 0)),
        'repair_begin': list(np.full(Num_nodes, 0))}
    )
    evo_conditions = monte_carlo_sequential(nodes_info, sample=False, T=T)
    return evo_conditions


if __name__ == '__main__':
    print('hello world!')

    # 完成网络的代码调试
    import_file = False
    Node_num = 100
    Topology = 'Random'
    Area_size = (250, 150)
    Area_width, Area_length = 250, 150
    Coordinates = generate_positions(Node_num, Area_width, Area_length)

    # TX_range = 50 # 传输范围为区域面积的1/5时能够保证网络全联通
    transmit_power = 15  # 发射功率(毫瓦)，统一单位：W
    path_loss = 2.5  # 单位：无
    noise = pow(10,
                -10)  # 噪声的功率谱密度(毫瓦/赫兹)，统一单位：W/Hz, 参考自https://dsp.stackexchange.com/questions/13127/snr-calculation-with-noise-spectral-density
    bandwidth = 20 * pow(10, 6)  # 带宽(Mhz)，统一单位：Hz
    lambda_TH = 8 * pow(10, -1)  # 接收器的敏感性阈值,用于确定节点的传输范围
    TX_range = pow((transmit_power / (bandwidth * noise * lambda_TH)), 1 / path_loss)
    CV_range = 30  # 节点的覆盖范围

    # 业务请求的参数
    App_num = 20
    grid_size = 5
    traffic_th = 0.5  # 业务网格的流量阈值
    App_Demand = np.random.normal(loc=5, scale=1, size=App_num)  # 生成平均值为5，标准差为1的业务带宽请求的整体分布
    App_Priority = [1, 2, 3]
    ratio_str = 1  # 尽量分离和尽量重用的业务占比
    Strategy_P = ['Global'] * int(App_num * (1 - ratio_str))
    Strategy_S = ['Local'] * int(App_num * ratio_str)
    App_Strategy = Strategy_S + Strategy_P

    G = Network(Topology, Node_num, Coordinates, TX_range, transmit_power, bandwidth, path_loss, noise, import_file)
    G, Apps = init_func(G, Coordinates, Area_size, CV_range, grid_size, traffic_th, App_num, App_Demand, App_Priority,
                        App_Strategy)

    # 演化条件的参数
    T = 8760
    MTTF, MLife = 1000, 800
    MTTR = 2



    Evo_conditions = cond_func(G, MTTF, MLife, MTTR, T)
    print('网络演化态的数量为{}'.format(len(Evo_conditions)))

