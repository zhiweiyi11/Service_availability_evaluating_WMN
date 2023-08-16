#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   ：Service_availability -> Network_evolultion_model
@IDE    ：PyCharm
@Author ：Yi Zhiwei
@Date   ：2023/7/31 10:20
@Desc   ：网络演化模型的主函数

=================================================='''
from typing import Any

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from Evolution_Objects import *
from Evolution_Conditions import *
from Evolution_Rules import *

def create_mapping_relationship(G, Apps):
    # storage the node and link information of service mapping
    link_info = {}
    for edge in G.edges:
        cap = G.edges[edge[0]][edge[1]]['capacity']
        link_info[edge] = {'curr_bw':cap, 'mapped_service':[]}

    for app_id in Apps.keys():
        service_path = Apps[app_id].path
        service_demand = Apps[app_id].demand
        for i in range(len(service_path)-1):
            # Check if the service path is valid
            if not nx.has_path(G, service_path[0], service_path[-1]):
                print("Invalid service path. No path exists between source and destination.")
            else:
                curr_node = service_path[i]
                next_node = service_path[i + 1]
                edge = (curr_node, next_node)
                link_info[edge]['mapped_service'].append(app_id)
                link_info[edge]['curr_bw'] -= service_demand

    return link_info

def state_transition(G, Apps, Evo_conditions: pd.DataFrame):
    # 在演化条件激励下的网络及服务状态转变
    time_list = Evo_conditions.index.tolist()
    fault_detection_rate = 0.9
    time_path_cal, time_process, time_propagation = 0.5, 0.05, 0.005 # 单位为s

    # 创建”网络演化态“
    link_info = create_mapping_relationship(G, Apps)

    App_interrupt = [] #发生中断的业务集合

    for i in range(len(time_list)):
        evo_time = time_list[i]
        fail_nodes = Evo_conditions.iloc[i]['fail']
        reco_nodes = Evo_conditions.iloc[i]['repair']

        # action1: 对应构件修复下的状态转变规则
        for node in reco_nodes:
            # 对网络和服务的状态进行更新
            component_repair(node, G)
        # action2: 对仍然中断的业务进行重路由计算(计算K最短路候选+带宽资源匹配策略)
        Reroute_successful = []
        for app_id in App_interrupt:
            # 进行重路由计算
            service = Apps[app_id]
            app_demand = service.demand
            new_path = k_shortest_paths()

            if new_path: # 如果重路由计算成功
                Reroute_successful.append(app_id)
                service.fail_time = 0 # 将服务故障时刻置零,表示恢复上线
                # service_remapping_path(app_id, app_demand, new_path, link_info)
                service.app_deploy_node(G)
                service.app_deploy_edge(G)
                statis_outage_duration(service, evo_time) # 统计服务中断的时长
                App_interrupt.remove(app_id)
            else:
                continue

        # action3: 对构件故障下的状态转变规则
        App_faulty = []
        for node in fail_nodes:
            faulty_links = component_fail(node, G)
            faulty_services = service_faulty_index(faulty_links, link_info)
            for service_id in faulty_services:
                if service_id not in App_faulty: # 去除重复统计的服务id
                    App_faulty.append(service_id)
                    Apps[service_id].fail_time = evo_time  # 将此时服务的中断时刻更新
                    Apps[service_id].app_undeploy_node(G)
                    Apps[service_id].app_undeploy_edge(G)

        # action4: 对故障的服务进行恢复检测
        App_reroute, App_repair = service_fault_detect(fault_detection_rate, App_faulty)
        App_interrupt += App_repair# 将故障检测不成功的业务加入中断的业务集合

        # action5: 对业务进行按优先级排序
        App_hash = {Apps[a].id: Apps[a].SLA for a in App_reroute}
        App_reroute_sorted = sorted(App_hash)
        current_faulty_service = len(App_reroute_sorted)
        for app_id in App_reroute_sorted:
            # 进行重路由计算
            service = Apps[app_id]
            app_demand = service.demand
            new_path = service_reroute(service, G, fail_nodes)
            if new_path: # 如果重路由计算成功
                Reroute_successful.append(app_id)
                service.fail_time = 0 # 将服务故障时刻置零,表示恢复上线
                # service_remapping_path(app_id, app_demand, new_path, link_info)
                service.app_deploy_node(G)
                service.app_deploy_edge(G)
                statis_recovery_duration(service, current_faulty_service, redirecting_node, merging_node, new_path_length,time_path_cal, time_process, time_propagation) # 统计服务中断的时长
                App_interrupt.remove(app_id)
            else:
                continue


def component_repair(node_id, G:nx.Graph):
    # 对构件进行修复的规则
    G.nodes[node_id]['alive'] = 1  # 置节点的状态为1，表示存活
    adj_nodes = list(G.adj[node_id])
    neighbors_id = G.neighbors(node_id)
    for adj in adj_nodes:
        G.adj[node_id][adj]['weight'] = 1 - G.adj[node_id][adj]['fail_rate']

def component_fail(node_id, G:nx.Graph):
    # 对构件进行故障注入
    faulty_links = []
    G.nodes[node_id]['alive'] = 0
    neighbors_id = G.neighbors(node_id)
    for nbr in neighbors_id:
        G.adj[node_id][nbr]['weight'] = float('inf')
        faulty_links.append((node_id, nbr))

    return faulty_links

def service_reroute(service:App, G, faulty_node, link_info):
    # 重路由的规则(k-最短路路由计算+路由匹配：first-fit or best-fit)
    source_node_lst = service.access
    destination_node_lst = service.exit
    required_bandwidth = service.demand
    original_path = service.path
    strategy = service.str
    redirecting_node, merging_node = None, None
    path = None

    if strategy == 'Global':
        # 找到重路由发起的源宿节点
        faulty_index = original_path.index(faulty_node[0])
        for u,v in zip(original_path, original_path[1:]):
            G[u][v]['weight'] = float('inf') # 将原路径置为无穷大，避免计算到原路径上去
        if faulty_index == 0 or faulty_index == len(original_path)-1:
            # 如果故障节点为source 或 target，则直接随机选择另外一个节点对
            new_source = choose_random_element(source_node_lst, faulty_node[0])
            new_target = destination_node

    for shortest_path in nx.all_shortest_paths(G, source_node, destination_node, 'weight'):
        # Check if the path has enough available bandwidth (first-fit strategy)
        available_path_bandwidth = min(G[u][v]['capacity'] - G[u][v]['load'] for u, v in zip(shortest_path, shortest_path[1:]))

        if available_path_bandwidth >= required_bandwidth:
            path = shortest_path
            break

    return path, redirecting_node, merging_node

def k_shortest_path_match(G, required_bw, source_node, destination_node):
    # 计算符合带宽需求的K最短路
    K = 5
    path = None
    path_lst = list(islice(nx.shortest_simple_paths(G, source, target, weight), K))
    for shortest_path in path_lst:
        available_path_bandwidth = min(G[u][v]['bandwidth'])


def choose_random_element(lst, specific_element):
    eligible_elements = [element for element in lst if element != specific_element]
    random_element = random.choice(eligible_elements)
    return random_element

def service_fault_detect(detection_rate, faulty_service_list):
    # 判断服务故障是否被成功检测
    successful_app_list = []  # 故障检测成功的服务列表
    unsuccessful_app_list = []
    # 生成一组随机数
    random_number_list = np.random.rand(len(failed_service_list))
    for i in range(len(random_number_list)):
        random_number = random_number_list[i]
        if random_number < detection_rate:
            successful_app_list.append(faulty_service_list[i])
        else:
            unsuccessful_app_list.append(faulty_service_list[i])

    return successful_app_list, unsuccessful_app_list


def service_remapping_path(app_id, app_demand, new_path, network_link_info):
    # 将服务的路径映射到对应的网络链路上
    for i in range(len(new_path) - 1):
        # Check if the service path is valid
        if not nx.has_path(G, new_path[0], new_path[-1]):
            print("Invalid service path. No path exists between source and destination.")
        else:
            curr_node = new_path[i]
            next_node = new_path[i + 1]
            edge = (curr_node, next_node)
            network_link_info[edge]['mapped_service'].append(app_id)
            network_link_info[edge]['curr_bw'] -= app_demand

def service_faulty_index(faulty_links, link_info):
    # 根据故障的链路索引发生故障的业务
    faulty_services = []
    for link in faulty_links:
        if link in link_info.keys():
            faulty_services += link_info[link]['mapped_service']
        else:
            faulty_services += link_info[link[::-1]]['mapped_service'] # 翻转link中的元素来进行索引
    faulty_services_final = list(set(faulty_services)) # 去除重复的业务id

    return faulty_services_final

def statis_outage_duration(service, evo_time):
    # 统计服务中断的时长
    last_fail_time = service.fail_time
    repair_duration = last_fail_time - evo_time
    service.outage['outage'].append(repair_duration)

def service_recovery_duration(service, current_service_num, redirecting_node, merging_node, new_path_length,time_path_cal, time_process, time_propagation):
    # 计算服务在恢复过程中的时长
    disruption_time = 0 # 服务重路由计算的时长
    degradation_time = 0 # 服务故障定位以及新路径响应的时长
    strategy = service.str # 服务的重路由策略

    disruption_time = current_service_num*time_path_cal + 2 * new_path_length * time_propagation
    if strategy == 'Global':
        degradation_time = (current_service_num + redirecting_node) * time_process + (redirecting_node+new_path_length) * time_propagation
        service.outage['reroute'].append(disruption_time)
        service.outage['degradation'].append(degradation_time)
    else:
        degradation_time = current_service_num*time_process + (merging_node+new_path_length) * time_propagation
        service.outage['reroute'].append(disruption_time)
        service.outage['degradation'].append(degradation_time)









if __name__ == '__main__':
    print('hello world')
