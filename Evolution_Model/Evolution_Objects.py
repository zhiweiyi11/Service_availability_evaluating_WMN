#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   ：Service_availability -> Network_Evolution_Objects
@IDE    ：PyCharm
@Author ：Yi Zhiwei
@Date   ：2023/3/6 14:36
@Desc   ：建模无线网络演化对象
=================================================='''
import copy
import os.path

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import random
import math
import time
import datetime
import re as replace

import pandas as pd

from Evolution_Model.Application_request_generating import *
import PIL # 导入计算机网络的图标

from Evolution_Model.Application_request_generating import *
from Evolution_Model.Network_position_generating import *
from Evolution_Model.Evolution_Rules import k_shortest_paths
# from sympy import *
from sympy import exp, symbols, diff

def calculate_distance(node1, node2):
  """Calculate the Euclidean distance between two nodes."""
  x1 = node1[0] # pos_x
  y1 = node1[1]
  x2 = node2[0]
  y2 = node2[1]
  return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

def saveDataFrameToExcel(df: pd.DataFrame, fileName: str):
    time = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M')
    # df.to_excel(r'..\Results_Saving\{}_time_{}.xlsx'.format(fileName, time), index=False)
    df.to_excel(r'..\Results_Saving\{}.xlsx'.format(fileName), index=False)

    print('成功保存数据文件 \n')


def generate_positions(NB_nodes, Area_width, Area_length, save_data):
    # 根据区域范围，生成节点的坐标
    # save_data = False
    pos_x = np.random.uniform(0, Area_width, NB_nodes)
    pos_y = np.random.uniform(0, Area_length, NB_nodes)
    # 将横纵坐标进行合成为array, 生成节点坐标的阵列
    Node_coordinates = np.column_stack((pos_x, pos_y))
    # sink_coordinates = np.array([[cf.BS_POS_X, cf.BS_POS_Y]])  # 第1个节点为sink节点,加2个[]是为了和sensor类型的array维度保持一致
    # 创建网络所有节点的坐标，其中sink节点在第1个
    # network_coordinates = np.concatenate((sink_coordinates, sensor_coordinates),axis=0)
    # 保存网络节点的坐标数据为excel
    df = pd.DataFrame(Node_coordinates)
    # Plotting
    # plt.scatter(pos_x, pos_y, edgecolor='b', facecolor='none', alpha=0.5)
    # plt.xlabel("x")
    # plt.ylabel("y")
    # plt.show()

    # time2 = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M')  # 记录数据存储的时间
    if save_data == True:
        path2 = os.path.abspath('..') # 表示当前所处的文件夹上一级文件夹的绝对路径
        with pd.ExcelWriter(path2 + r'.\Results_Saving\Node_Coordinates_{}_Uniform.xlsx'.format(NB_nodes)) as writer:
            df.to_excel(writer, sheet_name='Node_Coordinates')
            print("数据成功保存")

    return Node_coordinates

def function_SINR(s, d_ij, d_interferer_list, noise_power):
    m = 4 # 衰落系数的形状参数
    gamma = 1 # SINR阈值
    W = noise_power # 0.0001
    # d_ij = 20
    # d_interferer_list = [35]
    alpha = 2
    Pt = 1.5
    # s = symbols('s', real=True)
    k = (m * gamma * W) / (Pt*pow(d_ij, -alpha))

    y1 = 1
    for d_vj in d_interferer_list:
        b = gamma * pow((d_vj / d_ij), -alpha)
        y1 *= pow( (1+s*b), -m)
    y2 = exp(-k*s)  * y1

    return y2


def calculate_SINR_outage_probability(m, d_ij, d_interferer_list, noise_power):
    '''
    # 计算基于SINR的链路故障率
    :param gamma: SINR 阈值, 通常为1
    :param W / noise_power: 噪声功率, 0.0001 W
    :param alpha: 路损指数, 在[2,6]之间取值
    :param m: 衰落系数的形状参数, 4
    :param d_ij: 链路(v_i, v_j)的距离
    :param d_interferer_list: 干扰节点的距离集合
    :return:
    # 链路的中断概率主要受到3个参数的影响:噪声功率、路损指数、干扰链路的数量和距离
    # 当不存在干扰节点时,主要是噪声功率和路损指数影响;当前考虑干扰节点时, 主要受到干扰节点的距离影响,而且当干扰链路越多 或 距离<链路本身的距离时,链路故障率越高(0.99)
    '''
    res = 0
    s = symbols('s', real= True)
    for n in range(m):
        aa = pow(-1, n)/np.math.factorial(n)
        # print('aa is {}'.format(aa))
        bb = diff(function_SINR(s, d_ij, d_interferer_list, noise_power), s , n).subs({s:1})
        # print('bb is {}'.format(bb))
        res += aa*bb
    outage_probability = 1- res

    return outage_probability


class Network(nx.Graph): # 表示继承为nx.Graph的子类
    ''' 存储所有的节点及拓扑信息，以及对节点进行操作的方法'''

    def __init__(self, topology, TX_prob, Coordinates, TX_range, transmit_power, bandwidth, path_loss, noise, import_file, file_name):
        self.transmit_power = transmit_power # 节点的发射功率
        self.path_loss = path_loss
        self.noise = noise
        self.bandwidth = bandwidth
        self.transmit_prob = TX_prob
        self.node_neighbours = {} # 网络节点的邻居节点集合
        self.node_neighbours_dis = {} # 网络节点到邻居节点的距离集合

        if topology == 'Tree': # 按星型拓扑部署节点
            nx.Graph.__init__(self)
            r = 3
            h = 3
            G =  nx.balanced_tree(r, h) # r:each node will have r children; h: height of tree
            self.update(G)

        if topology == 'Random_SINR': # 随机部署节点
            # 创建二维网络（mesh）图
            nx.Graph.__init__(self)
            G = nx.Graph()

            edges_list = []  # 用于保存相连的（或具有连边的）节点对

            # 如果是从文件中导入节点坐标
            if import_file == True:
                # sys_path = os.path.abspath('..')  # 表示当前所处文件夹上一级文件夹的绝对路径
                Edges_info = pd.read_excel( r"..\Results_Saving\{}.xlsx".format(file_name), sheet_name=0, index_col=0)  # 读取第1个sheet的数据,不把表格的行索引读入进来
                print('读取网络链路的数据成功')
                ################### 这部分需要根据导入的数据格式进行完善 ###################
                for i in range(len(Edges_info.index)):
                    edge_str = Edges_info.index[i]
                    edge = replace.findall(r'\d+', edge_str) # 采用正则表达式，将df中读取的链路节点对的字符串中的数字提取出来
                    edge = list(map(int, edge))
                    distance = Edges_info.iloc[i]['distance']
                    capacity = Edges_info.iloc[i]['capacity']
                    fail_rate = Edges_info.iloc[i]['fail_rate']
                    weight = 1 - fail_rate

                    G.add_edge(*edge, dis=distance, capacity=capacity,load=0, fail_rate=fail_rate,  app_dp=[], weight=weight)
                self.update(G)
                self.set_node_attributes(Coordinates)  # 设置节点的属性

            else:  # 基于SINR来生成网络链路
                nx.Graph.__init__(self)
                G = nx.Graph()
                edges_list, edges_dis = [], []
                large_link_num = 0
                for i in range(len(Coordinates)):
                    neighbours, neighbours_dis = [], []  # 存储各节点的邻居节点以及对应距离
                    for j in range(len(Coordinates)):
                        if j == i:  # 避免将节点到自己的距离计算进去
                            continue
                        else:
                            distance = self.nodes_distance(Coordinates[i], Coordinates[j])
                            if distance <= TX_range: # 修改这里的判断
                                edges_list.append((i, j))
                                edges_dis.append(distance)
                                neighbours.append(j)  # 将节点i加入至其邻居节点中
                                neighbours_dis.append(distance)
                    self.node_neighbours[i] = neighbours
                    if len(neighbours) > 10:
                        large_link_num += 1
                        print('邻居节点个数大于10的链路为{}'.format(edges_list[-1]))
                    self.node_neighbours_dis[i] = neighbours_dis
                print('邻居节点个数大于10的链路总数为{} \n'.format(large_link_num))

                for k in range(len(edges_list)):  # 生成网络拓扑图
                    noise_power = self.bandwidth * self.noise
                    print('噪声功率为{}'.format(noise_power))
                    d_ij = edges_dis[k]
                    ts, rc = edges_list[k]  # 找出链路的节点对
                    SINR_ij, interfere_node_ij = self.calculate_SINR(ts, rc, self.transmit_prob)
                    SINR_ji, interfere_node_ji = self.calculate_SINR(rc, ts, self.transmit_prob)
                    capacity = 1 / 2 * (self.bandwidth * math.log((1 + SINR_ij), 2) + self.bandwidth * math.log((1 + SINR_ji),2)) / 1000000  # 转换为Mbps
                    fail_rate_ij = calculate_SINR_outage_probability(4, d_ij, interfere_node_ij, noise_power)
                    fail_rate_ji = calculate_SINR_outage_probability(4, d_ij, interfere_node_ji, noise_power)
                    fail_rate = 1 / 2 * (fail_rate_ij + fail_rate_ji)
                    print('链路的中断概率为{}\n'.format(fail_rate))
                    weight = 1 - fail_rate
                    G.add_edge(*edges_list[k], dis=d_ij, capacity=capacity, load=0, fail_rate=fail_rate, app_dp=[], weight=weight)  # 设置链路故障率为距离的函数

                self.update(G)
                self.set_node_attributes(Coordinates)  # 设置节点的属性

        # elif topology =='Random': # 如果为随机拓扑的话，则从节点坐标中生成网络链路
        #     nx.Graph.__init__(self)
        #     G = nx.Graph()
        #     # 计算小于节点范围内的连接链路
        #     edges_list, edges_dis = [], []
        #     for i in range(len(Coordinates)):
        #         for j in range(i + 1, len(Coordinates)):
        #             distance = self.nodes_distance(Coordinates[i], Coordinates[j])
        #             if distance <= TX_range:
        #                 edges_list.append((i, j))
        #                 edges_dis.append(distance)
        #
        #     for k in range(len(edges_list)):  # 生成网络拓扑图
        #         d = edges_dis[k]
        #         fail_rate = self.link_failure_probability(d, r_th=50, phi=1.5)  # r_th, phi的取值越大，失效率越低, d越大，故障率越高
        #         capacity = self.link_capacity(d)
        #         # weight = self.link_weight(fail_rate, 0, capacity)  # 初始时各链路上的负载为0
        #         weight = 1- fail_rate # 假设链路的权重为 1- 故障率
        #         G.add_edge(*edges_list[k], dis=d, capacity=capacity, load=0, fail_rate=fail_rate,  app_dp=[], weight=weight)  # 设置链路故障率为距离的函数
        #
        #     self.update(G)
        #     self.set_node_attributes(Coordinates)  # 设置节点的属性



    def calculate_SINR(self, transmit_node_id, receive_node_id, transmit_prob):
        # 计算链路的SINR
        interference = 0

        index = self.node_neighbours[transmit_node_id].index(receive_node_id)
        distance = self.node_neighbours_dis[transmit_node_id][index] # 获取节点对之间的距离
        print('链路的距离为{}'.format(distance))

        interfereNodeIdList = self.node_neighbours[receive_node_id]
        interfereDistanceList = self.node_neighbours_dis[receive_node_id]

        ''' 这里基于二项分布生成实际的干扰节点集合,即干扰节点的数量为一个随机变量'''
        # random_list = np.random.binomial(1, transmit_prob, len(interfereNodeIdList))
        # 干扰节点总数量为二项分布的期望值, 具体的节点为从干扰节点集合中随机选取 N_I个节点
        num_I = int((len(interfereNodeIdList)-1)*transmit_prob) # 干扰节点的数量需要减去当前的接收节点
        current_interfere_node = random.sample(interfereNodeIdList, num_I)
        print('当前的干扰节点为{} '.format(current_interfere_node))

        current_interfere_dis = [] # 当前实际的干扰节点的距离
        for i in current_interfere_node:
            if i == transmit_node_id:
                current_interfere_node.append(random.choice(interfereNodeIdList)) # 再从干扰节点list中随机选取一个加入到干扰节点集合中来
                continue
            else:
                index = interfereNodeIdList.index(i)
                current_interfere_dis.append(interfereDistanceList[index])

        print('当前的干扰链路距离为{} '.format(current_interfere_dis))


        for i in range(len(current_interfere_dis)): # 计算干扰的总功率
            interference += self.transmit_power * pow(interfereDistanceList[i], -self.path_loss)

        Pr = self.transmit_power * pow(distance, -self.path_loss) # 节点的接收功率

        sum_ = self.noise * self.bandwidth + interference # 这里主要考虑了噪声的功率谱密度

        SINR = Pr / sum_
        # print('链路{}的SINR为{}'.format((transmit_node_id, receive_node_id),SINR))

        return SINR, current_interfere_dis

    def link_failure_probability(self, r, r_th, phi):
        # 计算链路的失效概率
        # r_th is given by the same threshold as for the area mean power model
        # Empirically phi may vary in the [0,6]
        p = 1 / 2 * (1 + math.erf((10 * np.log(r / r_th) / (np.sqrt(2) * np.log(10) * phi))))
        return p
    def link_capacity(self, distance):
        signal_power = self.transmit_power * pow(distance, -self.path_loss)
        noise_power = self.noise * self.bandwidth
        transmit_rate = self.bandwidth * math.log((1+signal_power/noise_power), 2) / 1000000 # 转换为Mbps
        return transmit_rate

    def link_weight(self, link_fail_rate, link_load, link_capacity):
        # 计算链路的权重，使得路由可以找到更可靠的链路
        weight = (1-link_fail_rate)*(1- link_load/link_capacity)
        return weight

    def nodes_distance(self, node1, node2):
        """Calculate the Euclidean distance between two nodes."""
        x1 = node1[0]  # pos_x
        y1 = node1[1]
        x2 = node2[0]
        y2 = node2[1]
        return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    def set_node_attributes(self,  Coordi):
        # 设置节点上的属性值，其中参数值均为链路容量、MTTF 和 MTTR 的集合
        for n in list(self):
            node_attrs = {n: {'alive': 1,'pos': Coordi[n], 'app_dp':[] }}
            nx.set_node_attributes(self, node_attrs)

    def generate_link_state(self):
        # 根据各链路的故障率，生成链路的状态
        edges_list = list(self.edges)
        random_number_list = np.random.rand(len(edges_list)) # 根据链路的数量生成一个随机数组
        failed_link_num = 0 # 统计当前演化下故障的链路总数
        for i in range(len(random_number_list)):
            edge = edges_list[i]
            random_number = random_number_list[i]
            link_fail_rate = self.adj[edge[0]][edge[1]]['fail_rate']
            link_weight = self.adj[edge[0]][edge[1]]['weight']
            if random_number < link_fail_rate: # 此时链路发生了随机故障
                self.adj[edge[0]][edge[1]]['weight'] = float('inf')
                failed_link_num += 1
                # print('链路{}发生了故障'.format(edge))
            elif link_weight == float('inf'): # 如果原链路本来就中断了,则仍然设置其链路权重为inf
                self.adj[edge[0]][edge[1]]['weight'] = float('inf')
            else:
                self.adj[edge[0]][edge[1]]['weight'] = 1
        # print('当前演化态下链路中断的总数为{}'.format(failed_link_num))

    def restore_link_state(self) :
        # 在重路由计算结束后还原链路状态,即将各链路的权重还原为1
        nodes_list = list(self)
        for n in nodes_list:
            node_state = self.nodes[n]['alive']
            if node_state != 0: # 如果节点未故障的话
                adj_nodes = list(self.adj[n])
                for adj in adj_nodes:   # 将未failure的节点的邻接链路的权重恢复为非中断率
                    link_fail_rate = self.adj[n][adj]['fail_rate']
                    self.adj[n][adj]['weight'] = 1 - link_fail_rate
        # print('链路权重恢复正常 \n')


    def draw_topo(self, coordinates):
        # 根据节点的坐标绘制网络的拓扑结构图
        # pos = nx.spring_layout(self)
        nodes_list = list(self)  # 生成节点的列表
        position = {i: coordinates[i] for i in nodes_list}
        options = {
            'node_color': 'darkgreen',
            'edge_color': 'dimgrey',
            'node_size': 50,
            'alpha': 0.8,  # 节点和边的透明度
            'width': 0.8,  # 边的线宽
            'with_labels': False,  # 显示节点编号
        }
        plt.figure(figsize=(10, 7.5))  # 或者 fig, ax = plt.subplots()
        nx.draw_networkx_edges(self, position, alpha=0.4)
        nx.draw_networkx(self, position, **options)
        # plt.savefig(r'.\Pictures_saved\Topology_of_network_{}_nodes.jpg'.format(len(self.nodes)), dpi= 1200)
        plt.show()

class App(object):
    # 定义一个业务对象的类，保存网络的业务逻辑层信息
    def __init__(self, id, source, target, path, demand, priority, strategy):
        # 以下是人为给定的
        self.id = id # 业务的识别号，字符串表示
        self.access = source # 业务请求的接入节点
        self.exit = target # 业务请求的终端节点
        self.demand = demand # 业务的带宽需求
        self.SLA = priority # 业务的优先级
        self.str = strategy # 业务的重路由策略

        # 以下是根据网络的拓扑和链路带宽资源分布生成的
        self.path = path # 业务部署的路径集合
        self.load = demand # 记录业务路径重部署后的实际负载值（假设初始业务负载 ==  业务的带宽需求）


        self.outage = {'reroute':[], 'repair':[], 'degradation':[] } # 业务的中断时刻(重路由、修复以及降级)
        self.fail_time = 0 # 业务故障的时刻（初始值为0）
        self.down_time = 0 # 业务性能降级的时刻

    def app_deploy_edge(self, G):
        # 业务至基础设施网络的映射

        # 由于业务存在多条路径所以需要对每条路径上业务的传输流量进行确定
        for i in range(len(self.path) - 1):  # 依次读取路径列表中前后2个数据组成连边
            if G.has_edge(self.path[i], self.path[i + 1]):  # 判断业务路径是否有真实的物理拓扑
                app_list = G.get_edge_data(self.path[i], self.path[i + 1])['app_dp']  # 将该边上原有的、已加载业务信息读取出来，因为后面更新改变的业务属性时会被覆盖掉
                if self.id in app_list: # 如果出现了重复的id，表明业务已经部署在链路上，此时仅调整网络链路的负载
                    print('重复部署的 app id is {}'.format(self.id))
                    print('业务的路径为{}, 当前的链路为{}'.format(self.path, [self.path[i], self.path[i+1]]))
                else:
                    app_list.append(self.id)  # 元组中前面的表示业务的id，后面的表示业务对应的子路径index
                    G.adj[self.path[i]][self.path[i + 1]]['app_dp'] = app_list  # 更新链路的业务部署信息
                    G.adj[self.path[i]][self.path[i + 1]]['load'] += self.load # 仅将业务当前的负载加至链路上
                    available_cap = G.adj[self.path[i]][self.path[i + 1]]['capacity'] - G.adj[self.path[i]][self.path[i + 1]]['load']
                    if available_cap < 0:
                        print('当前业务id {}的负载{}部署导致链路{}过载,链路的信息为{}'.format(self.id,self.load, (self.path[i], self.path[i+1]), G.adj[self.path[i]][self.path[i + 1]]))
            else:
                break

    def app_undeploy_edge(self, G):
        for i in range(len(self.path) - 1):  # 依次读取路径列表中前后2个数据组成连边
            if G.has_edge(self.path[i], self.path[i + 1]):  # 判断业务路径是否有真实的物理拓扑
                app_list = G.get_edge_data(self.path[i], self.path[i + 1])['app_dp']  # 将该边上原有的、已加载业务信息读取出来，因为后面更新改变的业务属性时会被覆盖掉
                if self.id not in app_list:
                    print('edge is {}, and info is {}'.format([self.path[i], self.path[i+1]], G.edges[self.path[i], self.path[i+1]] ))
                    print('app id is {}'.format(self.id))
                    print('app path is {}'.format(self.path))
                # else:
                #     print('app {} is successfully removed '.format(self.id))
                app_list.remove(self.id)
                G.adj[self.path[i]][self.path[i + 1]]['app_dp'] = app_list  # 更新链路的业务部署信息
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
                else:
                    print('当前业务卸载时节点 {}不存在'.format(i))
                    break
        else:
            print('业务待卸载{}的路径是空集'.format(self.id))


def generate_app_path(G, source, destination, app_demand):
    # 根据业务的源宿节点和业务请求计算服务带宽的业务路径
    K = 10
    G_sample = copy.deepcopy(G)
    bottleneck_link = []
    app_path = []
    # print('重路由计算得到的候选路径集为{}'.format(new_subpath_optional))
    while K > 0:
        optional_path = nx.shortest_path(G_sample, source, destination, 'weight')

        new_path, bottleneck_link = link_capacity_allocation(G_sample, optional_path, app_demand)
        if new_path and len(new_path)<10: # 确保首次生成的业务路径长度小于10跳
            app_path = new_path
            break
        else:
            K -= 1
            for link in bottleneck_link:
                G_sample.adj[link[0]][link[1]]['weight'] = float('inf')

    return app_path


def link_capacity_allocation(G,  candidate_path, app_demand):
    '''
    # 网络的链路带宽分配规则
    :param G: 当前网络链路的剩余带宽情况
    :param app_original_path: 业务原有的路径集合
    :param path_list: 业务的候选路径集（根据K-最短路算法计算而来）
    :param band_demand: 业务的带宽需求
    :return: 业务分配的新路径
    '''
    app_path = []
    link_load_path = [] # 记录业务候选路径上的链路剩余可用容量
    bottleneck_link = []
    for i in range(len(candidate_path)-1):
        link_available_cap = G.adj[candidate_path[i]][candidate_path[i+1]]['capacity'] - G.adj[candidate_path[i]][candidate_path[i+1]]['load']
        if link_available_cap < app_demand:
            link = (candidate_path[i], candidate_path[i+1])
            bottleneck_link.append(link)
            print('瓶颈链路为{},剩余带宽为{}'.format(link, link_available_cap))
        link_load_path.append(link_available_cap)

    capacity_available = min(link_load_path) # 计算业务路径上的最小带宽

    if capacity_available >= app_demand:
        app_path = candidate_path # 如果当前链路满足业务的带宽需求，则将该路径选择为业务的新路径

    return app_path, bottleneck_link

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



def init_func(G, Coordinates, Area_size, CV_range , grid_size,  traffic_th, App_num, App_Demand, App_Priority, App_Strategy):
    # 初始化网络演化对象函数，分别返回“基础设施对象”和“业务逻辑对象”
    # 根据基础设施网络来生成业务层对象
    Node_Coordinates = {} # 存储节点的坐标信息
    for n in list(G):
        Node_Coordinates[n] = Coordinates[n]


    Area_width, Area_length = Area_size[0], Area_size[1]
    App_dict = {}
    random.shuffle(App_Priority) # 将制定业务等级的list打乱，但是各等级的数量仍然不变
    TrafficDensity = generateAppTraffic(Area_width, Area_length, grid_size, traffic_th)
    App_coordinates, App_traffic = generateAppOD_from_grid(TrafficDensity[0], App_num) # 根据各网格上的流量密度和业务请求的数量,找到业务请求对应的od和demand
    df = pd.DataFrame(columns=['coordinates', 'demand'])
    # 将业务流量请求的数据保存下来
    for i in range(len(App_traffic)):
        coord = App_coordinates[i]
        demand = App_traffic[i]
        df.loc[i] = [coord, demand]
    # saveDataFrameToExcel(df, 'GridTraffic_[request={}]'.format(App_num))

    for i in range(len(App_coordinates)):
        #　随机选择2个不重复的网格节点来作为业务的od对
        app_od_coord = random.sample(TrafficDensity[0].keys(), 2)
        # od_list = []
        while True:
            print('计算业务的接入和接出节点\n')
            App_coordinates[i] = app_od_coord
            App_access = access_mapping(Node_Coordinates, App_coordinates, CV_range)
            access = App_access[0][i]
            exit = App_access[1][i]
            if access and exit: # 如果业务可接入节点和可退出节点都不为空的话
                od_list = RecursionFunc(access, [exit])
                break
            else:
                app_od_coord = random.sample(TrafficDensity[0].keys(), 2)

        priority = App_Priority[i]
        # demand = App_traffic[i] # 业务的流量从密度矩阵中生成
        demand = App_Demand[i] # 业务的请求的流量从外部输入
        strategy = App_Strategy[i]
        # source, destination = od_list[0][0], od_list[0][1]
        # tmp_od = od_list[0]

        while True:
            # print('业务的od列表为{} \n'.format(od_list))
            # if od_list:
            tmp_od = random.choice(od_list)  # 从业务可接入节点集合中随机选择一个作为其源节点
            source = tmp_od[0]
            destination = tmp_od[1]
            print('计算业务的初始部署路径 \n')

            # 加入约束，保证业务路径是多跳的(业务的接入和退出节点不能相同)
            # path_tmp = nx.shortest_path(G, source, destination, 'weight') # 找包含业务源宿节点图G中一条最短路径作为业务的初始路径
            # app_path= link_capacity_allocation(G, path_tmp, demand) # 根据链路的容量来确定业务是否部署在该路径上
            app_path = generate_app_path(G, source, destination, demand)
            if app_path:
                app = App(i, access, exit, app_path, demand, priority, strategy)
                print('业务{}的初始路径为{}'.format(i, app_path))
                app.app_deploy_node(G) # 将业务的id映射至节点
                app.app_deploy_edge(G) # 将业务的流量映射至链路
                App_dict[i] = app
                break
            if od_list: # 如果业务的od节点可选集合不为空
                od_list.remove(tmp_od) # 删除掉不满足业务请求的od对
                print('删除的业务{}的od对为{},剩余的od对列表长度为{}'.format(i, tmp_od, len(od_list)))
                if od_list: # 如果业务的od对列表为空则跳出循环
                    print('第{}个业务初始路径部署失败...'.format(i))
                    break


    return G, App_dict



def load_AppInfoFromExcel(fileName: str) -> dict:
    """
    从磁盘导入Excel表格并读取其中的业务信息
    :param fileName: 导入文件的文件名
    :return: 业务字典
    """
    AppDict = {}

    df = pd.read_excel(io='..\Results_Saving\{}.xlsx'.format(fileName))
    for i in range(len(df)):
        id = df.loc[i]['id']

        access = df.loc[i]['access']
        access = str(access) + ' '
        access = access.split() # 将access字符串转换为单个字符组成的列表
        access_list = list(map(int, access)) # 将读入的字符串列表转换为int格式的列表

        exit = df.loc[i]['exit']
        exit = str(exit) + ' '
        exit = exit.split()
        exit_list = list(map(int, exit))

        demand = df.loc[i]['demand']
        SLA = df.loc[i]['SLA']
        path = df.loc[i]['path']
        path = path.split()
        path_list = list(map(int, path))
        strategy = df.loc[i]['strategy']

        app = App(id, access_list, exit_list, path_list, demand, SLA, strategy)
        AppDict[id] = app
    print('成功从excel文件中导入业务请求数据 \n')

    return AppDict

def init_function_from_file(Topology_file, Coord_file, AppDict_file, Network_parameters, Wireless_parameters, Loss_parameters):
    # 从文件中初始化网络和业务对象
    Node_coord = pd.read_excel(io= "..\Results_Saving\{}.xlsx".format(Coord_file), sheet_name=0, index_col=0)  # 读取第1个sheet的数据,不把表格的行索引读入进来
    print('读取节点坐标数据成功')
    Coordinates = {}
    for n in Node_coord.index: # 将dataframe中存储的节点坐标转换为 dict格式
        Coordinates[n] = Node_coord.loc[n].to_list()

    Topology, TX_prob = Network_parameters[0], Network_parameters[1]
    TX_range, transmit_power, bandwidth = Wireless_parameters[0], Wireless_parameters[1], Wireless_parameters[2]
    path_loss, noise = Loss_parameters[0], Loss_parameters[1]
    G = Network(Topology, TX_prob, Coordinates, TX_range, transmit_power, bandwidth, path_loss, noise, True, Topology_file)
    # G.draw_topo(Coordinates)

    AppDict = load_AppInfoFromExcel(AppDict_file)

    for app_id, app_object in AppDict.items():
        # 依次将业务部署至网络上去
        app_object.app_deploy_node(G)  # 将业务的id映射至节点
        app_object.app_deploy_edge(G)  # 将业务的流量映射至链路

    return G, AppDict

if __name__ == '__main__':
    # 参考现有的WSN仿真器的代码，完成WSN网络的构建
    # 完成网络的代码调试
    save_data = True # 不保存节点坐标数据
    Node_num =  100
    Topology = 'Random_SINR'
    Area_size = (150, 150)
    # Area_width, Area_length = 500,500# 250, 200
    # Coordinates = generate_positions(Node_num, Area_size[0], Area_size[1], save_data)
    # Coordinates_sample = generate_PPP_distribution(Area_size, Node_num, save_data) # 当节点传输半径较小时，通过增加节点的数量来保证网络的连通性
    Coordinates_file =  'Node_Coordinates_100_Uniform[for_priority_analysis]' # 'Different_resourceAndDemand_Topology=100+App=50\\Node_Coordinates_100_Uniform'
    Coordinates_df = pd.read_excel( r"..\Results_Saving\{}.xlsx".format(Coordinates_file), sheet_name=0, index_col=0)
    Coordinates = dict(zip(Coordinates_df.index, Coordinates_df.values))

    # TX_range = 50 # 传输范围为区域面积的1/5时能够保证网络全联通
    transmit_prob = 0.1 # 节点的数据发送概率
    transmit_power = 1.5  # 发射功率(毫瓦)，统一单位：W
    path_loss = 2  # 单位：无
    noise = pow(10, -11)  # 噪声的功率谱密度(毫瓦/赫兹)，统一单位：W/Hz, 参考自https://dsp.stackexchange.com/questions/13127/snr-calculation-with-noise-spectral-density
    Band = 10
    bandwidth = Band * pow(10, 6)  # 带宽(Mhz)，统一单位：Hz 10* pow(10, 6) 改变网络的容量和带宽;
    lambda_TH = 8 * pow(1, -1)  # 接收器的敏感性阈值,用于确定节点的传输范围
    # TX_range = pow((transmit_power / (bandwidth * noise * lambda_TH)), 1 / path_loss) # 传输范围为38.8
    TX_range = 30 # 为了让整网链路的平均故障率低，需要将节点的传输范围设置较小值,并且增加节点的密度保证网络的拓扑连通性
    CV_range = 30 # 节点的覆盖范围

    # 业务请求的参数
    App_num = 50
    grid_size = 5
    traffic_th = 1 # 业务网格的流量阈值
    Demand = 5
    App_Demand_generation = np.random.normal(loc=Demand, scale=1, size=App_num) # 生成平均值为3，标准差为1的业务带宽请求的整体分布
    App_Demand = [] # 确保生成的所有业务需求值为正
    for d in App_Demand_generation:
        if d > 0:
            App_Demand.append(d)
        else:
            App_Demand.append(-d)

    App_Priority = [1,2,3,4,5] * int(App_num / 5) # 用于分析不同业务优先级对服务可用度的影响，假设各等级的业务数量相同
    ratio_str = 1 # 尽量分离和尽量重用的业务占比
    Strategy_P = ['Global'] * int(App_num*(1-ratio_str))
    Strategy_S = ['Local'] * int(App_num*ratio_str)
    App_Strategy = Strategy_S + Strategy_P

    # 确定是否从文件导入数据
    import_topology_file = True # 不从文件中导入拓扑数据
    file_name = 'Topology_100_Band=10[for_priority_analysis]' # 'Different_resourceAndDemand_Topology=100+App=50\\Topology_100_Band=20'

    ## 这是初始随机生成网络及业务对象的代码
    G = Network(Topology, transmit_prob, Coordinates, TX_range, transmit_power, bandwidth, path_loss, noise, import_topology_file, file_name)
    G, Apps = init_func(G, Coordinates, Area_size, CV_range,  grid_size, traffic_th, App_num, App_Demand, App_Priority, App_Strategy)
    # #
    Edges_info = {}
    edges_list = list(G.edges)
    for i in range(len(edges_list)):
        edge_info = []
        e = edges_list[i]
        edge_info.append(e)
        edge_info.append(G.edges[e[0], e[1]]['dis'])
        edge_info.append(G.edges[e[0], e[1]]['capacity'])
        edge_info.append(G.edges[e[0], e[1]]['fail_rate'])
        Edges_info[i] = edge_info

    # save_GraphInfo(Edges_info, 'Topology_{}_Band={}'.format(Node_num, Band) )
    save_AppInfo(Apps, 'App_{}_Demand={}_inTopo={}_Band={}'.format(App_num, Demand, Node_num, Band))
    # # # Apps_load = load_AppInfoFromExcel('App_100_SINR')
    # #
    # # # # 保存网络拓扑和业务请求的数据至Excel
    # df_coord = pd.DataFrame(Coordinates_sample)
    # with pd.ExcelWriter(r'..\Results_Saving\Node_Coordinates_{}_SINR.xlsx'.format(Node_num)) as writer:
    #     df_coord.T.to_excel(writer, sheet_name='Node_Coordinates')
    #     print("数据成功保存")
    #
    #
    # ''' 这里的代码用来测试从excel文件中导入'''
    Network_parameters = [Topology, transmit_prob]
    Wireless_parameters = [TX_range, transmit_power, bandwidth]
    Loss_parameters = [path_loss, noise]

    # G, Apps = init_function_from_file('Topology_150_SINR', 'Node_Coordinates_150_SINR','App_150_SINR', Network_parameters, Wireless_parameters, Loss_parameters)
    G.draw_topo(Coordinates)

    ave_link_fail = 0
    num_efficient_link = 0
    for e in G.edges:
        link_fail = G.adj[e[0]][e[1]]['fail_rate']
        ave_link_fail += link_fail
        if link_fail < 0.3:
            num_efficient_link += 1
        # print('链路{}的故障率为{}'.format(e, link_fail))
    ave_link_fail = ave_link_fail/len(G.edges)

    print('整网链路的平均故障率为{}'.format(ave_link_fail))
    print('网络的链路总数为{},有效的链路数为{}'.format(len(G.edges), num_efficient_link))

