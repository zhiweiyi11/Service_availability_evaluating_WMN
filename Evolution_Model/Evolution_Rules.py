#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   ：Service_availability -> Evolution_Rules
@IDE    ：PyCharm
@Author ：Yi Zhiwei
@Date   ：2023/3/6 16:50
@Desc   ：网络构件故障后业务重路由的规则
=================================================='''
import random

import networkx as nx
import numpy as np
import copy
from itertools import islice # 创建一个迭代器，返回从 iterable 里选中的元素
from Evolution_Model.Evolution_Objects import *
from Evolution_Model.Evolution_Conditions import cond_func
from Evolution_Model.Application_request_generating import *

def k_shortest_paths(k, G, source, target,  weight=None):
    # use this function to efficiently compute the k shortest/best paths between two nodes
    return list(islice(nx.shortest_simple_paths(G, source, target, weight=weight), k))

# 路由规则
def path_reroute(G, app_access, app_exit, app_path, app_demand, app_strategy, node_fail_list, routing_th):
    '''
    # 网络的重路由规则
    :param G: 当前网络的拓扑结构
    :param app_access, app_exit: 业务接入和接出的节点集合
    :param path: 业务当前部署的路径
    :param node_fail_list: 故障模式（故障链路的集合）
    :parm strategy: 重路由策略（尽量重用 or 尽量分离）
    :return: 业务的新路径
    '''
    G_sample = copy.deepcopy(G)
    reroute_times = 0 # 重路由次数的计数
    new_app_path = [] # 业务重路由的候选路径集合
    source = app_path[0]
    destination = app_path[-1]

    '''业务的重路由分为2种策略：Global和Local'''
    if app_strategy == 'Global':
        # 全局重路由策略(从源宿节点重新找一条路)
        # print('业务的重路由策略为{}'.format(app_strategy))

        # 1) 先将原始的路径上的链路权重设置为无穷大
        for i in range(len(app_path)-1):
            G_sample.adj[app_path[i]][app_path[i+1]]['weight'] = float('inf')
        # 2) 然后在业务的接入和接出节点集合中寻找一条满足业务带宽需求的路径
        for node in node_fail_list:
            fail_node_index = app_path.index(node)
            app_access_tmp = copy.deepcopy(app_access)
            app_exit_tmp = copy.deepcopy(app_exit)
            ## 确定业务重路由的源宿节点
            if fail_node_index == 0:  # 如果故障的节点是链路的首节点或尾节点
                # 根据节点的状态来重新选择业务的源节点
                if len(app_access_tmp) > 1:  # 如果业务有其余可以接入的节点list
                    available_access_list = []  # 存储可接入节点的list
                    for n in app_access:
                        if G_sample.nodes[n]['alive'] == 1:
                            available_access_list.append(n)
                        else:
                            app_access_tmp.remove(n)
                    if available_access_list:  # 如果存在可接入的节点list
                        source = random.choice(available_access_list)
                    else:  # 如果业务仅1个接入节点且发生了故障，则直接返回空的路径
                        return new_app_path, reroute_times
                else:
                    return new_app_path, reroute_times

            elif fail_node_index == len(app_path) - 1:
                if len(app_exit_tmp) > 1:
                    available_exit_list = []
                    for n in app_exit:
                        if G_sample.nodes[n]['alive'] == 1:
                            available_exit_list.append(n)
                        else:
                            app_exit_tmp.remove(n)
                    if available_exit_list:  # 如果存在可接入的节点list
                        source = random.choice(available_exit_list)
                    else:  # 如果业务仅1个接入节点且发生了故障，则直接返回空的路径
                        return new_app_path, reroute_times

                else:
                    return new_app_path, reroute_times
            else: # 如果故障的节点为业务的中继节点，则直接从其源宿节点list中随机选择一个进行重路由
                source = random.choice(app_access_tmp)
                destination = random.choice(app_exit_tmp)

    '''业务的Local重路由策略'''
    if app_strategy == 'Local':
    # 找出节点故障导致的业务原路径中故障的链路
    #     print('业务的重路由策略为{}'.format(app_strategy))
        for node in node_fail_list:
            fail_node_index = app_path.index(node)
            app_access_tmp = copy.deepcopy(app_access)
            app_exit_tmp = copy.deepcopy(app_exit)
            ## 确定业务重路由的源宿节点
            if fail_node_index == 0 : # 如果故障的节点是链路的首节点或尾节点
                # 根据节点的状态来重新选择业务的源节点
                if len(app_access_tmp) > 1: # 如果业务有其余可以接入的节点list
                    available_access_list = [] # 存储可接入节点的list
                    for n in app_access:
                        if G_sample.nodes[n]['alive'] == 1:
                            available_access_list.append(n)
                        else:
                            app_access_tmp.remove(n)
                    if available_access_list: # 如果存在可接入的节点list
                        source = random.choice(available_access_list)
                    else: # 如果业务仅1个接入节点且发生了故障，则直接返回空的路径
                        return new_app_path, reroute_times
                else:
                    return new_app_path, reroute_times

            elif fail_node_index == len(app_path)-1:
                # fail_link = (app_path[fail_node_index-1], node)
                # link_fail_list.append(fail_link)
                # 重新选择业务的宿节点
                if len(app_exit_tmp)>1:
                    available_exit_list = []
                    for n in app_exit:
                        if G_sample.nodes[n]['alive'] == 1:
                            available_exit_list.append(n)
                        else:
                            app_exit_tmp.remove(n)
                    if available_exit_list:  # 如果存在可接入的节点list
                        source = random.choice(available_exit_list)
                    else:  # 如果业务仅1个接入节点且发生了故障，则直接返回空的路径
                        return new_app_path, reroute_times
                else:
                    return new_app_path, reroute_times


            else: # 如果业务的故障模式为中继节点
                # fail_link_1, fail_link_2 = (node, app_path[fail_node_index+1]), (app_path[fail_node_index-1], node)
                # link_fail_list.append(fail_link_1)
                # link_fail_list.append(fail_link_2)
                source = app_path[fail_node_index - 1]
                destination = app_path[fail_node_index + 1]

    ''' 
    根据业务的故障模式(源宿节点故障or中继节点故障)，对业务进行重路由
    '''

    path = nx.shortest_path(G_sample, source, destination, 'weight')
    while len(path) > 0: # 如果路径非空
    # 计算路径的存在概率
        path_exist = 0
        for i in range(len(path) - 1):
            delta = random.random()
            link_fail_rate = G_sample.adj[path[i]][path[i + 1]]['fail_rate']
            cap_available = min( G_sample.nodes[path[i]]['cap'], G_sample.nodes[path[i+1]]['cap'])
            # print('生成的随机数为{}'.format(delta))
            # print('链路的故障率为{}\n'.format(link_fail_rate))

            if delta < link_fail_rate or cap_available < app_demand:  # 如果随机数小于该链路的中断率，则该链路不存在，需要重新选路
                G_sample.adj[path[i]][path[i + 1]]['weight'] = float('inf') # 置当前链路的权重为无穷大
                if reroute_times > routing_th:
                    # 如果重路由次数超过了业务重路由的阈值
                    new_app_path = []
                    reroute_times = 0
                    break
                else:
                    # 重新计算路由
                    reroute_times += 1
                    break
            else:# 链路存在的索引值+1
                path_exist += 1

        if path_exist == len(path)-1:
            new_app_path = path
            break
        else:
            path = nx.shortest_path(G_sample, source, destination, 'weight')


    return new_app_path, reroute_times



# 资源（带宽）分配规则
def resource_assign(G, path_list, app_demand):
    '''
    # 网络的带宽分配规则：First-Fit算法（将业务的候选路径集按路径长度从小到大排序，依次遍历各路径上的剩余带宽，如果满足业务带宽需求，则更新为业务的路径）
    :param G: 当前网络链路的剩余带宽情况
    :param app_original_path: 业务原有的路径集合
    :param path_list: 业务的候选路径集（根据K-最短路算法计算而来）
    :param band_demand: 业务的带宽需求
    :return: 业务分配的新路径
    '''
    app_path = []
    bottle_neck_link = []# 记录导致业务倒换失败的瓶颈链路
    for path in path_list:
        # band_available = 200
        length = sum(G.adj[path[i]][path[i+1]]['weight'] for i in range(len(path)-1))
        if length == float('inf'): # 如果链路长度为无穷大
            continue
        else:
            least_link_band = []
            for i in range(len(path)-1):
                least_band = G.adj[path[i]][path[i+1]]['cap'] - G.adj[path[i]][path[i+1]]['load']
                least_link_band.append(least_band)
            band_available = min(least_link_band) # 计算业务路径上的最小带宽

            if band_available >= app_demand:
                app_path = path # 如果当前链路满足业务的带宽需求，则将该路径选择为业务的新路径
                break
            else:
                id = least_link_band.index(band_available) # 根据瓶颈带宽的值索引到对应的链路id
                edge = (path[id],path[id+1])
                if edge not in bottle_neck_link: # 避免重复加入瓶颈链路
                    bottle_neck_link.append(edge)
        # else:
        #     print('链路的剩余可用带宽为{}'.format(band_available)) # 检测业务的最小可用带宽是否为负数

    return app_path


if __name__ == '__main__':
    # 调试尽量分离的重路由算法；
    Topology = 'Random'
    MTTF, MTTR, Cap = 400, 4, 5
    Node_num, App_num = 100, 30
    Capacity = 50
    Demand = np.random.normal(loc=10, scale=2, size=App_num)  # 生成平均值为5，标准差为1的带宽的整体分布
    Area_width , Area_length = 250, 150
    Area_size = (250,150)


    TX_range = 50  # 传输范围为区域面积的1/5时能够保证网络全联通
    CV_range = 30
    Coordinates = generate_positions(Node_num, Area_width, Area_length)
    # Demand = list(map(int, Demand)) # 将业务的带宽需求换成整数
    grid_size = 5
    traffic_th = 0.5
    Priority = np.linspace(start=1, stop=5, num=5, dtype=int)
    ratio_str = 1  # 尽量分离和尽量重用的业务占比
    Strategy_P = ['Global'] * int(App_num * (1 - ratio_str))
    Strategy_S = ['Local'] * int(App_num * ratio_str)
    Strategy = Strategy_S + Strategy_P

    G, App = init_func(Area_size, Node_num, Topology, TX_range, CV_range, Coordinates, Capacity, grid_size,  App_num, traffic_th, Demand, Priority, Strategy)  # 这样子的测试需要在测试函数中返回输出结果才行

    app_path = App[15].path
    app_access = App[15].access
    app_exit = App[15].exit
    app_demand = App[15].demand
    node_fail_list = [app_path[0]]

    for n in node_fail_list:
        G.nodes[n]['alive'] = 0  # 置节点的状态为1，表示存活
        # 将节点相邻节点的链路权重设置为1，表示节点恢复上线
        adj_nodes = list(G.adj[n])
        for adj in adj_nodes:
            G.adj[n][adj]['weight'] = float('inf')

    candi_path, reroute_times = path_reroute(G, app_access, app_exit, app_path, app_demand, 'Global' , node_fail_list, 10)
