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

def save_GraphInfo(link_dict:dict, filename:str):
    # 导出网络的拓扑信息{链路：节点对，故障率，容量}
    df = pd.DataFrame(columns=['node_pair', 'distance', 'capacity', 'fail_rate'])
    for i in range(len(link_dict)):
        node_pair = link_dict[i][0] # 因为存储整网链路的dict中各属性是存储在一个list中
        distance = link_dict[i][1]
        capacity = link_dict[i][2]
        fail_rate = link_dict[i][3]

        df.loc[i] = [node_pair, distance,  capacity, fail_rate]

    saveDataFrameToExcel(df, filename)


def generate_PPP_distribution(area_size,  lam, save_data):
    '''
    :param area_size: (xMax, yMax)长方形区域的尺寸
    :param lam: # intensity (ie mean density) of the Poisson process,表示网络的期望节点数
    :return:
    '''
    # 生成2维的长方形区域内的泊松点分布
    xMax, xMin = 1, 0 # 区域大小先归1化处理
    yMax, yMin = 1, 0
    xDelta = xMax - xMin
    yDelta = yMax - yMin # rectangle dimensions
    areaTotal = xDelta * yDelta

    Coordinates = {} # 将节点的坐标存储为dict格式

    # Simulate Poisson point process
    numbPoints = np.random.poisson(lam * areaTotal)  # Poisson number of points
    pos_x = xDelta * np.random.uniform(0, 1, numbPoints) + xMin  # x coordinates of Poisson points
    pos_y = yDelta * np.random.uniform(0, 1, numbPoints) + yMin  # y coordinates of Poisson points
    Node_coordinates = np.column_stack((pos_x*area_size[0], pos_y*area_size[1])) # 将横纵坐标分别乘以区域的长和宽放大
    for i in range(len(Node_coordinates)):
        Coordinates[i] = Node_coordinates[i]

    positions = dict(zip(range(len(Node_coordinates)), Node_coordinates)) # 生成网络节点的位置,为一个dict类型

    # Plotting
    plt.scatter(pos_x, pos_y, edgecolor='b', facecolor='none', alpha=0.5)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

    if save_data == True:
        df = pd.DataFrame(positions)
        with pd.ExcelWriter( r'..\Results_Saving\Node_Coordinates_{}_PPP.xlsx'.format(lam)) as writer:
            df.T.to_excel(writer, sheet_name='Node_Coordinates')
            print("数据成功保存")

    return Coordinates

def generate_Mesh_topo(num_nodes):
    # 生成MESH拓扑的网络
    G = nx.grid_2d_graph(num_nodes, num_nodes)  # 5x5 grid

    # print the adjacency list
    for line in nx.generate_adjlist(G):
        print(line)
    # write edgelist to grid.edgelist
    nx.write_edgelist(G, path="grid.edgelist", delimiter=":")
    # read edgelist from grid.edgelist
    H = nx.read_edgelist(path="grid.edgelist", delimiter=":")

    pos = nx.spring_layout(H, seed=200)
    nx.draw(H, pos)
    plt.show()

    return pos, G

def generate_mesh(width, height, unit):
    """
    将一个矩形区域划分为更细致的网格
    :param width: 区域的宽度
    :param height: 区域的高度
    :param unit: 网格边长，假设以正方形为最小单位进行划分
    :return: 节点坐标
    """
    x = []
    y = []
    res = []

    for i in range(0, width + unit, unit):
        x.append(i)

    for i in range(0, height + unit, unit):
        y.append(i)

    for i in range(0, len(x)):
        for j in range(0, len(y)):
            temp = []
            temp.append(x[i])
            temp.append(y[j])
            res.append(temp)

    return res

def get_edges_from_mesh(coordinates, unit) -> []:
    """
    将mesh中相邻的坐标形成边
    :param coordinates: mesh坐标集合
    :param unit: mesh的最小放个边长
    :return: 具有连边的节点的序号对
    """
    edges_list = []

    for i in range(len(coordinates)):
        for j in range(len(coordinates)):
            distance = calculate_distance(coordinates[i], coordinates[j])
            if (distance == unit):
                edges_list.append((i, j))

    for i in range(len(edges_list)):
        for j in range(i + 1, len(edges_list)):
            x_i = edges_list[i][0]
            y_i = edges_list[i][1]
            x_j = edges_list[j][0]
            y_j = edges_list[j][1]
            if (x_i == y_j and y_i == x_j):
                del edges_list[j]
                break

    return edges_list


def generate_tree_topo(num_nodes):
    # 生成树状拓扑
    tree = nx.random_tree(n=num_nodes, seed=10)
    pos = nx.spring_layout(tree)
    nx.draw(tree, pos)
    plt.show()

    return tree, pos

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
    # save_AppInfo(Apps, 'App_100_randomTopo')
    area_size = [150, 150]
    num_nodes = 10
    # Coordinates_sample = generate_PPP_distribution(area_size, num_nodes)
    Coord = generate_mesh(150, 150, 15)
    G = generate_Mesh_topo(num_nodes)
    # G = nx.grid_2d_graph(Coord)
