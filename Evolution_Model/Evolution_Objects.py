#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   ：Service_availability -> Network_Evolution_Objects
@IDE    ：PyCharm
@Author ：Yi Zhiwei
@Date   ：2023/3/6 14:36
@Desc   ：建模无线网络演化对象
=================================================='''
import os.path

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import random
import math
import time
import datetime

import pandas as pd

from Evolution_Model.Application_request_generating import *
import PIL # 导入计算机网络的图标

from Evolution_Model.Application_request_generating import *


def calculate_distance(node1, node2):
  """Calculate the Euclidean distance between two nodes."""
  x1 = node1[0] # pos_x
  y1 = node1[1]
  x2 = node2[0]
  y2 = node2[1]
  return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def generate_positions(NB_nodes, Area_width, Area_length):
    # 根据区域范围，生成节点的坐标
    save_data = False
    pos_x = np.random.uniform(0, Area_width, NB_nodes)
    pos_y = np.random.uniform(0, Area_length, NB_nodes)
    # 将横纵坐标进行合成为array, 生成节点坐标的阵列
    sensor_coordinates = np.column_stack((pos_x, pos_y))
    # sink_coordinates = np.array([[cf.BS_POS_X, cf.BS_POS_Y]])  # 第1个节点为sink节点,加2个[]是为了和sensor类型的array维度保持一致
    # 创建网络所有节点的坐标，其中sink节点在第1个
    # network_coordinates = np.concatenate((sink_coordinates, sensor_coordinates),axis=0)
    # 保存网络节点的坐标数据为excel
    df = pd.DataFrame(sensor_coordinates)
    time2 = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M')  # 记录数据存储的时间
    if save_data == True:
        path2 = os.path.abspath('..') # 表示当前所处的文件夹上一级文件夹的绝对路径
        with pd.ExcelWriter(path2 + r'.\Results_Saving\Node_Coordinates_created_from_{}.xlsx'.format(time2)) as writer:
            df.to_excel(writer, sheet_name='Node_Coordinates')
            print("数据成功保存")
    return sensor_coordinates


def cal_link_fail_probability(r, r_th, phi):
    # 计算链路的失效概率
    # r_th is given by the same threshold as for the area mean power model
    # Empirically phi may vary in the [0,6]
    p = 1/2 *(1+math.erf( (10*np.log(r/r_th) / (np.sqrt(2)*np.log(10)*phi) )))
    return p

class Network(nx.Graph): # 表示继承为nx.Graph的子类
    ''' 存储所有的节点及拓扑信息，以及对节点进行操作的方法'''

    def __init__(self, topology, NB_NODES, Coordinates, TX_range, import_file):
        if topology == 'Star': # 按星型拓扑部署节点
            nx.Graph.__init__(self)
            G =  nx.star_graph(NB_NODES+1)
            self.update(G)

        if topology == 'Grid': # 随机部署节点
            # 创建二维网络（mesh）图
            nx.Graph.__init__(self)
            m = int(np.sqrt(NB_NODES))
            n = m + 2
            G = nx.grid_2d_graph(m, n)  # 生成一个m行n列的方格网络
            self.update(G)

        if topology =='Random':
            nx.Graph.__init__(self)
            G = nx.Graph()
            # 计算小于节点范围内的连接链路
            edges_list, edges_dis = [], []
            if import_file == True:# 如果从文件中导入节点坐标
                sys_path = os.path.abspath('..') # 表示当前所处文件夹上一级文件夹的绝对路径
                Node_coord = pd.read_excel(io= sys_path + r".\Results_Saving\Node_Coordinates.xlsx", sheet_name=0, index_col=0) # 读取第1个sheet的数据,不把表格的行索引读入进来
                print('读取节点坐标数据成功')

                for i in range(len(Node_coord)):
                    for j in range(i + 1, len(Node_coord)):
                        distance = calculate_distance(Node_coord.loc[i], Node_coord.loc[j])
                        if distance <= TX_range:
                            edges_list.append((i,j))
                            edges_dis.append(distance)

                for k in range(len(edges_list)):
                    d = edges_dis[k]
                    fail_rate = cal_link_fail_probability(d, r_th=50, phi=1.5) # r_th, phi的取值越大，失效率越低, d越大，故障率越高
                    G.add_edge(*edges_list[k], dis = d, fail_rate = fail_rate, load=0, weight= 1)  # 设置链路故障率为距离的函数
            else:
                for i in range(len(Coordinates)):
                    for j in range(i + 1, len(Coordinates)):
                        distance = calculate_distance(Coordinates[i], Coordinates[j])
                        if distance <= TX_range:
                            edges_list.append((i, j))
                            edges_dis.append(distance)

                for k in range(len(edges_list)):
                    d = edges_dis[k]
                    fail_rate = cal_link_fail_probability(d, r_th=50, phi=1.5)  # r_th, phi的取值越大，失效率越低, d越大，故障率越高
                    G.add_edge(*edges_list[k], dis=d, fail_rate=fail_rate, load=0, weight=1)  # 设置链路故障率为距离的函数

            self.update(G)

    def set_edge_attributes(self, Cap):
        # 设置边上的属性值，其中参数值均为链路容量、MTTF 和 MTTR 的集合
        for e in self.edges():
            i = 0
            edge_attrs = {e: {'cap': Cap[i]}}
            nx.set_edge_attributes(self, edge_attrs)
            i += 1

    def set_node_attributes(self,  Cap, Coordi):
        # 设置节点上的属性值，其中参数值均为链路容量、MTTF 和 MTTR 的集合
        for n in list(self):
            node_attrs = {n: {'alive': 1,'pos': Coordi[n],'cap':Cap, 'load':0, 'app_dp':[] }}
            nx.set_node_attributes(self, node_attrs)



    def draw_topo(self, coordinates, import_file):
        # 根据节点的坐标绘制网络的拓扑结构图
        # pos = nx.spring_layout(self)
        nodes_list = list(self) # 生成节点的列表
        # position = dict(zip(self.nodes, coordinates))
        if import_file == True:
            # 从导入的excel表格中读取节点坐标
            sys_path = os.path.abspath('..')  # 表示当前所处文件夹上一级文件夹的绝对路径
            Node_coord = pd.read_excel(io=sys_path + r".\Results_Saving\Node_Coordinates.xlsx", sheet_name=0,
                                       index_col=0)  # 读取第1个sheet的数据,不把表格的行索引读入进来

            position = {i: Node_coord.iloc[i] for i in nodes_list}
        else:
            position = {i: coordinates[i] for i in nodes_list}
        # print(pos)
        options = {
            'node_color': 'darkgreen',
            'edge_color': 'dimgrey',
            'node_size':50,
            'alpha':0.8, # 节点和边的透明度
            'width': 0.8, # 边的线宽
            'with_labels':False, # 显示节点编号
        }
        plt.figure(figsize=(10, 7.5)) # 或者 fig, ax = plt.subplots()

        # nx.draw_networkx(self,pos= position, **options)
        # nx.draw_networkx_nodes(self, pos = position, nodelist=[0], node_color='blue', node_shape='^', node_size=150)
        # plot with a nice basemap 在拓扑绘制的基础上加上底图https://networkx.org/documentation/stable/auto_examples/geospatial/plot_points.html
        """ # color by path length from node near center """ # https://networkx.org/documentation/latest/auto_examples/drawing/index.html
        # find node near center (125,125)
        dmin = 30
        node_center = 0
        for n in position:
            x, y = position[n]
            d = np.sqrt((x - 125) ** 2 + (y - 125) ** 2)
            if d < dmin:
                node_center = n
                dmin = d
        p = dict(nx.single_source_shortest_path_length(self, node_center)) # Compute the shortest path lengths from source to all reachable nodes.
        # p.pop(0) # 删除掉sink节点的坐标，以免重复绘制

        nx.draw_networkx_edges(self, position, alpha=0.4)
        nx.draw_networkx_nodes(
            self,
            position,
            nodelist=list(p.keys()),
            node_size=80,
            node_color=list(p.values()),
            cmap=plt.cm.Reds_r,
        )
        # plt.savefig(r'.\Pictures_saved\Topology_of_network_{}_nodes.jpg'.format(len(self.nodes)), dpi= 1200)

        plt.show()


class App(object):
    # 定义一个业务对象的类，保存网络的业务逻辑层信息
    def __init__(self,id, access_list, exit_list, band_demand, priority, path, strategy):
        self.id = id # 业务的识别号，字符串表示
        self.access = access_list # 业务请求的接入节点集合
        self.exit = exit_list # 业务请求的终端节点集合
        self.demand = band_demand # 业务的带宽需求
        self.SLA = priority # 业务的优先级
        self.str = strategy # 业务的重路由策略

        self.load = band_demand # 业务的实际带宽（假设初始业务负载 ==  业务的带宽需求）
        self.path = path # 业务部署的路径

        self.outage = {'reroute':[], 'repair':[]} # 业务的中断时刻
        self.fail_time= 0 # 业务故障的时刻（初始值为0）

    def app_deploy_edge(self, G):
        # 业务至基础设施网络的映射
        for i in range(len(self.path) - 1):  # 依次读取路径列表中前后2个数据组成连边
            if G.has_edge(self.path[i], self.path[i + 1]):  # 判断业务路径是否有真实的物理拓扑
                # app_list = G.get_edge_data(self.path[i], self.path[i + 1])['Apps']  # 将该边上原有的、已加载业务信息读取出来，因为后面更新改变的业务属性时会被覆盖掉
                # app_list.append(self.id)
                # G.adj[self.path[i]][self.path[i + 1]]['Apps'] = app_list # 更新链路的业务承载信息
                G.adj[self.path[i]][self.path[i + 1]]['load'] += self.load
            else:
                break

    def app_undeploy_edge(self, G): # 对边的处理仅考虑负载
        # 业务解除到基础设施网络的映射
        for i in range(len(self.path) - 1):  # 依次读取路径列表中前后2个数据组成连边
            if G.has_edge(self.path[i], self.path[i + 1]):  # 判断业务路径是否有真实的物理拓扑
                G.adj[self.path[i]][self.path[i + 1]]['load'] -= self.load # 链路的负载更新为 减去业务当前的负载
            else:
                break

    def app_deploy_node(self, G):
        # 根据业务的服务集合和路径，将业务映射到网络的节点上
        if self.path:
            # 如果业务路径非空，则对业务进行映射
            for i in self.path: # 从路径集合中读取业务流经的节点
                if G.has_node(i): #Graph.has_node(n)是方法，应该用()来表示，括号内的值为参数
                    app_list = G.nodes[i]['app_dp']
                    app_list.append(self.id)
                    G.nodes[i]['app_dp'] = app_list
                    G.nodes[i]['load'] += self.demand
                else:
                    print('当前业务部署时节点 {}不存在'.format(i))
                    break
        else:
            print('业务待部署{}的路径是空集'.format(self.id))

    def app_undeploy_node(self, G):
        # 根据业务的服务集合和路径，将业务映射到网络的节点上
        if self.path:
            # 如果业务路径非空，则对业务进行映射
            for i in self.path: # 从路径集合中读取业务流经的节点
                if G.has_node(i):
                    app_list = G.nodes[i]['app_dp']
                    app_list.remove(self.id) # 更新节点上部署的业务信息
                    G.nodes[i]['app_dp'] = app_list
                    G.nodes[i]['load'] -= self.demand
                else:
                    print('当前业务卸载时节点 {}不存在'.format(i))
                    break
        else:
            print('业务待卸载{}的路径是空集'.format(self.id))

def cap_assign(G,  path, app_demand):
    '''
    # 网络的链路带宽分配规则
    :param G: 当前网络链路的剩余带宽情况
    :param app_original_path: 业务原有的路径集合
    :param path_list: 业务的候选路径集（根据K-最短路算法计算而来）
    :param band_demand: 业务的带宽需求
    :return: 业务分配的新路径
    '''
    app_path = []

    band_available = min([ G.nodes[n]['cap'] ] for n in path) # 计算业务路径上的最小带宽

    if band_available >= app_demand:
        app_path = path # 如果当前链路满足业务的带宽需求，则将该路径选择为业务的新路径

    return app_path

def RecursionFunc(arr1,arrList):
    # 递归函数，从arrList中各取出一个元素，并进行组合
    if (arrList):
        string = []
        for x in arr1:
            for y in arrList[0]:
                if x != y: # 需要确保不会出现源宿节点重合
                    string.append( (x,y) ) # 从业务access和exit的list中分别取一个节点，组成业务请求的od
        result = RecursionFunc(string,arrList[1:])
        return result
    else:
        return arr1

def init_func(Area_size , Node_num, Topology, TX_range, CV_range, Coordinates, Capacity, grid_size, App_num, traffic_th,  Demand, Priority, Strategy):
    # 初始化网络演化对象函数，分别返回“基础设施对象”和“业务逻辑对象”
    # 首先生成基础设施网络
    Ifc = Network(Topology, Node_num, Coordinates, TX_range, True) #最后一个参数表示是否从文件中导入节点坐标
    Num_nodes = len(Ifc.nodes())
     # 根据坐标随机生成的网络拓扑，可能会导致图中的节点数小于坐标中的节点数量
    Node_Coordinates = {} # 存储节点的坐标信息
    for n in list(Ifc):
        Node_Coordinates[n] = Coordinates[n]
    Ifc.set_node_attributes(Capacity, Node_Coordinates) # 每个节点的容量相同
    # Ifc.set_edge_attributes(Cap_list)
    # Ifc.draw_topo(Coordinates, True) # 绘制网络的拓扑结构

    # 根据基础设施网络来生成业务层对象
    Area_width, Area_length = Area_size[0], Area_size[1]
    App_dict = {}
    random.shuffle(Priority) # 将制定业务等级的list打乱，但是各等级的数量仍然不变
    TrafficDensity = generateAppTraffic(Area_width, Area_length, grid_size, traffic_th)


    for i in range(App_num):
        #　随机选择2个不重复的网格节点来作为业务的od对
        app_od_coord = random.sample(TrafficDensity[0].keys(), 2)
        App_coordinates = {} # 存储业务坐标信息的字典
        od_list = []
        while True:
            App_coordinates[i] = app_od_coord
            App_access = access_mapping(Node_Coordinates, App_coordinates, CV_range)
            access = App_access[0][i]
            exit = App_access[1][i]
            if access and exit:
                od_list = RecursionFunc(access, [exit])
                break
            else:
                app_od_coord = random.sample(TrafficDensity[0].keys(), 2)

        pri = random.choice(Priority)
        demand = Demand[i]
        strategy = Strategy[i]

        while True:
            tmp_od = random.choice(od_list)  # 从业务可接入节点集合中随机选择一个作为其源节点
            source = tmp_od[0]
            destination = tmp_od[1]
            # 加入约束，保证业务路径是多跳的(业务的接入和退出节点不能相同)
            path_tmp = nx.shortest_path(Ifc, source, destination) # 找包含业务源宿节点图G中一条最短路径作为业务的初始路径
            app_path= cap_assign(Ifc, path_tmp, demand) # 根据链路的容量来确定业务是否部署在该路径上
            if app_path:
                app = App(i, access, exit, demand, pri, app_path, strategy)
                print('业务{}的初始路径为{}'.format(i, app_path))
                app.app_deploy_node(Ifc)
                App_dict[i] = app
                break
            elif od_list: # 如果业务的od节点可选集合不为空
                od_list.remove(tmp_od) # 删除掉不满足业务请求的od对
            else:
                print('第{}个业务初始路径部署失败...'.format(i))
                break
            # app = App(i, od, band_demand, pri, path_new, str)
            # app.app_deploy(Ifc)

    return Ifc, App_dict

if __name__ == '__main__':
    # 参考现有的WSN仿真器的代码，完成WSN网络的构建
    # 完成网络的代码调试
    Node_num =  100
    App_num = 20
    Cap_node = 20
    Cap_edge = 20
    Topology = 'Random'
    Area_size = (250,150)
    Area_width, Area_length = 250, 150

    TX_range = 50 # 传输范围为区域面积的1/5时能够保证网络全联通
    CV_range = 30
    Coordinates = generate_positions(Node_num, Area_width, Area_length)

    grid_size = 5
    traffic_th = 0.5 # 业务网格的流量阈值
    Demand = np.random.normal(loc= 5, scale=1, size=App_num) # 生成平均值为5，标准差为1的带宽的整体分布
    Priority = [1]*App_num
    ratio_str = 1 # 尽量分离和尽量重用的业务占比
    Strategy_P = ['Almost_Repetition'] * int(App_num*(1-ratio_str))
    Strategy_S = ['Almost_Separate'] * int(App_num*ratio_str)
    Strategy = Strategy_S + Strategy_P

    G, App_dict = init_func(Area_size, Node_num, Topology, TX_range, CV_range, Coordinates, Cap_node, grid_size, App_num, traffic_th, Demand, Priority, Strategy)
    # G.draw_topo(Coordinates)
    ave_link_fail = 0
    for e in G.edges:
        link_fail = G.adj[e[0]][e[1]]['fail_rate']
        ave_link_fail += link_fail
        # print('链路{}的故障率为{}'.format(e, link_fail))
    ave_link_fail = ave_link_fail/len(G.edges)

    print('整网链路的平均故障率为{}'.format(ave_link_fail))

