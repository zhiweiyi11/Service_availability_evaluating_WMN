#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   ：Service_availability -> Network_position_generating
@IDE    ：PyCharm
@Author ：Yi Zhiwei
@Date   ：2023/6/16 14:01
@Desc   ：根据不同的拓扑结构，来生成网络节点的坐标，以作为IWN网络对象的输入
=================================================='''

import random
import numpy as np
import networkx as nx
from matplotlib import pyplot as plt
import pandas as pd
import datetime
from Evolution_Model.Evolution_Objects import *

def generate_random_topology(Num_Nodes, dis, pos):

    # Use seed when creating the graph for reproducibility
    G = nx.random_geometric_graph(n= Num_Nodes, radius=dis, pos = pos, seed=896803) # n :  Number of nodes, radius: Distance threshold value
    # position is stored as node attribute data for random_geometric_graph
    pos = nx.get_node_attributes(G, "pos")
    plt.figure(figsize=(10, 7.5))
    nx.draw(G, pos)
    plt.show()
    return pos

def read_from_excel(file_name, ):
    #从excel中读取数据
    pass

def saveDataFrameToExcel(df: pd.DataFrame, fileName: str):
    time = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M')
    # df.to_excel(r'..\Results_Saving\{}_time_{}.xlsx'.format(fileName, time), index=False)
    df.to_excel(r'..\Results_Saving\{}.xlsx'.format(fileName), index=False)

    print('成功保存数据文件 \n')

def save_AppInfo(appDict: dict, fileName: str):
    """
    导出业务的详细信息
    :param appDict: 业务字典
    :param fileName: 导出后的文件名
    """
    df = pd.DataFrame(columns=['id', 'access', 'exit', 'demand', 'load', 'fail_time','down_time', 'SLA', 'path', 'strategy', 'outage'])
    for i in range(len(appDict)):
        app = appDict.get(i)
        id = app.id
        access = [str(i) for i in app.access] # 将业务的接入节点集合存储为字符串列表，便于后续读取
        access_str = ' '.join(access)

        exit = [str(j) for j in app.exit ]
        exit_str = ' '.join(exit)

        demand = app.demand
        load = app.load
        fail_time = app.fail_time
        down_time = app.down_time
        sla = app.SLA

        path = [str(k) for k in app.path ]
        path_str = ' '.join(path)

        strategy = app.str
        outage = app.outage

        df.loc[i] = [id, access_str, exit_str, demand, load, fail_time, down_time, sla, path_str, strategy, outage]

    saveDataFrameToExcel(df, fileName)







if __name__ == '__main__':
    # 测试networkx内置函数 随机地理图 的用法
    # N =100
    # dis = 0.15 # 传输范围为 250*dis=37.5
    # pos = {i: (random.gauss(0, 2), random.gauss(0, 2)) for i in range(N)} # 2D Gaussian distribution of node positions with mean (0, 0) and standard deviation 2
    # coord = generate_random_topology(N, dis, None) # None 表示节点为均匀分布
    # Coordinates = {}
    # for index, coordinates in coord.items():
    #     # 放大每个节点的坐标
    #     Coordinates[index] = [i * 250 for i in coordinates]
    # print('成功绘制了随机地理图')

    # 将Evolution_Objects 中初始化得到的网络和业务请求信息进行保存
    Apps = {}
    save_AppInfo(Apps, 'App_100_randomTopo')

