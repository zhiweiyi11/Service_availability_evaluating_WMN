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
def k_shortest_paths(k, G, source, target,  weight, strategy, original_path_length):
    # use this function to efficiently compute the k shortest/best paths between two nodes
    candidate_paths = []
    res = list(islice(nx.shortest_simple_paths(G, source, target, weight=weight), k)) # This procedure is based on algorithm by Jin Y. Yen [1]. Finding the first K paths requires O(K N^3) operations
    # print('计算得到的K最短路径集合为{}'.format(res))
    if strategy == 'Local':
        max_hop = 10
        for p in res:
            if len(p) <= max_hop:
                candidate_paths.append(p)
    else:
        max_hop = original_path_length + 5
        for p in res:
            if len(p) <= max_hop:
                candidate_paths.append(p)

    return candidate_paths



## 构件故障下的网络以及业务的状态转变
def component_failure(G, Apps, fail_time, failed_component_list):
    # 将故障节点所有邻接链路的权重置为inf
    apps_fault = []
    for n in failed_component_list:
        G.nodes[n]['alive'] = 0  # 先将节点的状态设置为0，表示此时节点故障
        failed_app_id = G.nodes[n]['app_dp']
        # print('节点{}上部署的业务为{}'.format(n, failed_app_id))
        apps_fault  += failed_app_id
        adj_nodes = list(G.adj[n])
        for adj in adj_nodes:
            G.adj[n][adj]['weight'] = float('inf')  # 将节点邻接的边的权重设置为无穷大

    apps_removed = [] # 存储一个待移除的业务id集合
    for app_id in apps_fault: #　避免将已经故障等待修复的业务加入重路由的业务集合中
        # print('节点{}故障下其上部署的业务为{}'.format(failed_component_list, apps_fault))
        if Apps[app_id].fail_time == 0: # 表明上一时刻业务未发生故障
            Apps[app_id].fail_time = fail_time # 记录故障发生的起始时刻
            # print('业务{}的故障时刻为{}'.format(app_id, fail_time))
        else:
            apps_removed.append(app_id) # 另一种操作逻辑是:当构件故障后,仅保留业务到该故障的构件映射，解除掉业务到其他健康构件的映射,这里需要进行app_undeploy_node的操作,同时,当业务修复后,再deploy回来
            # print('业务app {} 上一时刻{}已经发生了故障'.format(app_id, Apps[app_id].fail_time))
    # 最后统一移除节点的id(切记不能在for循环中对id进行移除)
    for id in apps_removed:
        apps_fault.remove(id)

    return apps_fault


## 构件修复下的网络及业务状态转变
def component_repair(G, Apps, App_interrupt, repair_time, repaired_component_list):
    # 将修复后的节点相邻的链路权重设置为1/权重系数

    for n in repaired_component_list:
        G.nodes[n]['alive'] = 1  # 置节点的状态为1，表示存活
        repaired_app_id = G.nodes[n]['app_dp'] # 读取存储在节点上的业务id
        # 将节点相邻节点的链路权重设置为1，表示节点恢复上线
        adj_nodes = list(G.adj[n])
        for adj in adj_nodes:
            G.adj[n][adj]['weight'] = 1 - G.adj[n][adj]['fail_rate']

        for app_id in repaired_app_id:
            if app_id in App_interrupt: # 如果业务id是属于中断的业务集合，则对该业务进行修复
                # print('完成修复的app为{} '.format(app_id))
                repair_duration = repair_time - Apps[app_id].fail_time
                if repair_duration > 50:
                    print('repair duration is {}'.format(repair_duration))
                    print('业务的故障时刻为{}，修复时刻为{}'.format( Apps[app_id].fail_time, repair_time))
                Apps[app_id].fail_time = 0  # 表示业务成功恢复上线
                Apps[app_id].outage['repair'].append(repair_duration)
                if Apps[app_id].down_time > 0 :# 若上一时刻业务处于降级状态(这种情况发生于故障检测不成功的业务)
                    load = Apps[app_id].load
                    duration = repair_time - Apps[app_id].down_time
                    Apps[app_id].down_time = 0
                    Apps[app_id].outage['degradation'].append({load:duration})

                # print('移除的业务id为{}'.format(app_id))
                App_interrupt.remove(app_id) # 可能是这里移除节点操作有问题(是前面component_failure函数中对一个正在遍历for循环的对象进行remove有问题)
                # print('中断的业务集合为{}'.format(App_interrupt))

    return App_interrupt



## 故障检测规则
def app_fault_detect(detection_successful_probability, failed_service_list):
    # 判断服务故障是否被成功检测
    successful_app_list = []  # 故障检测成功的服务列表
    unsuccessful_app_list = []
    # 生成一组随机数
    random_number_list = np.random.rand(len(failed_service_list))
    for i in range(len(random_number_list)):
        random_number = random_number_list[i]
        if random_number < detection_successful_probability:
            successful_app_list.append(failed_service_list[i])
        else:
            unsuccessful_app_list.append(failed_service_list[i])

    return successful_app_list, unsuccessful_app_list


# 路由规则
def path_reroute(G, app_demand, app_access, app_exit, app_path,  app_strategy, node_fail_list, recovery_parameters):
    '''
    # 网络的重路由规则
    :param G: 当前网络的拓扑结构
    :param app_access, app_exit: 业务接入和接出的节点集合
    :param path: 业务当前部署的路径
    :param node_fail_list: 故障模式（故障链路的集合）
    :parm strategy: 重路由策略（尽量重用 or 尽量分离）
    :return: 业务的新路径
    '''
    K = 5 # 表示K最短路径的数量
    G_sample = copy.deepcopy(G)
    # 根据recovery model计算重路由时长的参数
    message_processing_time, path_calculating_time, rerouting_app_num = recovery_parameters[0], recovery_parameters[1], recovery_parameters[2]
    time_propagation = 0.005
    new_app_path = [] # 业务重路由的候选路径集合
    new_app_load = 0
    reroute_duration = 0
    degradation_duration = 0
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
        faulty_node = node_fail_list[0] # 取出故障节点集合中的第一个节点
        fail_node_index = app_path.index(faulty_node)
        # 确保从未故障的节点集合中选择路由的源宿节点

        ## 确定业务重路由的源宿节点
        if fail_node_index == 0:  # 如果故障的节点是链路的首节点或尾节点
            # 根据节点的状态来重新选择业务的源节点
            if len(app_access) > 1:  # 如果业务有其余可以接入的节点list
                available_access_list = []  # 存储可接入节点的list
                for n in app_access:
                    if G_sample.nodes[n]['alive'] == 1:
                        available_access_list.append(n)
                if available_access_list:  # 如果存在可接入的节点list
                    source = random.choice(available_access_list)
                else:  # 如果业务仅1个接入节点且发生了故障，则直接返回空的路径
                    return new_app_path, reroute_duration, degradation_duration
            else:
                return new_app_path, reroute_duration, degradation_duration

        elif fail_node_index == len(app_path) - 1:
            if len(app_exit) > 1:
                available_exit_list = []
                for n in app_exit:
                    if G_sample.nodes[n]['alive'] == 1:
                        available_exit_list.append(n)
                if available_exit_list:  # 如果存在可接入的节点list
                    destination = random.choice(available_exit_list)
                else:  # 如果业务仅1个接入节点且发生了故障，则直接返回空的路径
                    return new_app_path, reroute_duration, degradation_duration

            else:
                return new_app_path, reroute_duration, degradation_duration

        else: # 如果故障的节点为业务的中继节点，则直接从其源宿节点list中随机选择一个进行重路由
            source = app_path[0]
            destination = app_path[-1]
        # 计算新的业务路径
        # new_app_path = nx.shortest_path(G_sample, source, destination, 'weight')
        original_path_length = len(app_path)
        new_app_path_optional = k_shortest_paths(K, G_sample, source, destination,  'weight', app_strategy, original_path_length)
        new_app_path = find_available_path(G_sample, app_path, app_demand, new_app_path_optional, faulty_node)
        if new_app_path: # 如果路径存在
            redirecting_node = fail_node_index # 从故障上游节点到计算恢复复路径source节点的跳数
            merging_node = 0
            reroute_duration, degradation_duration = service_recovery_duration(app_strategy, rerouting_app_num, redirecting_node, merging_node, len(new_app_path), path_calculating_time, message_processing_time, time_propagation)
            # print('业务重路由成功,原路径为{},新路径为{}'.format(app_path, new_app_path))
        else:
            print('业务重路由unsuccessful,原路径为{},候选路径为{}'.format(app_path, new_app_path_optional))



    '''业务的Local重路由策略'''
    if app_strategy == 'Local':
    # 找出节点故障导致的业务原路径中故障的链路
    #     print('业务的重路由策略为{}'.format(app_strategy))
        # 先将原路径上链路权重设置为inf，避免计算到的最短路径出现环路
        for i in range(len(app_path) - 1):
            G_sample.adj[app_path[i]][app_path[i + 1]]['weight'] = float('inf')

        faulty_node = node_fail_list[0]
        fail_node_index = app_path.index(faulty_node)
        ## 确定业务重路由的源宿节点
        if fail_node_index == 0 : # 如果故障的节点是链路的首节点或尾节点
            # 根据节点的状态来重新选择业务的源节点
            if len(app_access) > 1: # 如果业务有其余可以接入的节点list
                available_access_list = [] # 当前演化态下存储可接入节点的list
                for n in app_access:
                    if G_sample.nodes[n]['alive'] == 1 and n not in app_path:
                        available_access_list.append(n)
                if available_access_list: # 如果存在可接入的节点list
                    source = random.choice(available_access_list)
                    # destination = app_path[-1]
                    destination = app_path[fail_node_index+1]
                else: # 如果业务仅1个接入节点且发生了故障，则直接返回空的路径
                    return new_app_path, reroute_duration, degradation_duration
            else:
                return new_app_path, reroute_duration, degradation_duration

        elif fail_node_index == len(app_path)-1:

            # 重新选择业务的宿节点
            if len(app_exit) > 1:
                available_exit_list = []

                for n in app_exit:
                    if G_sample.nodes[n]['alive'] == 1 and n not in app_path: #需要确保业务重新选择的宿节点不能与原路径的节点相同
                        available_exit_list.append(n)
                if available_exit_list:  # 如果存在可接入的节点list
                    destination = random.choice(available_exit_list)
                    # source = app_path[0]
                    source = app_path[fail_node_index-1] # local策略下重路由时，若宿节点故障，发起重路由的源节点仍然不变

                else:  # 如果业务仅1个接入节点且发生了故障，则直接返回空的路径
                    return new_app_path, reroute_duration, degradation_duration
            else: # 如果业务的接出节点仅有1个
                return new_app_path, reroute_duration, degradation_duration
        else: # 如果业务的故障模式为中继节点
            source = app_path[fail_node_index - 1]
            destination = app_path[fail_node_index + 1]

        # # 在计算子路径之前，将现有的可用节点的相邻链路的权重设置为无穷
        # for n in app_path:
        #     if n != source and n!= destination: # G_sample.nodes[n]['alive'] == 1
        #         # 将除了源宿节点外的其他节点的相邻链路都设置为inf,避免路由计算中加入这些节点(还需要加入故障节点,因为主循环中对链路设置为inf可能没有传进来)
        #         adj_nodes = list(G_sample.adj[n])
        #         for adj in adj_nodes:
        #             G_sample.adj[n][adj]['weight'] = float('inf')  # 将节点邻接的边的权重设置为无穷大


        original_path_length = len(app_path)
        new_app_path, reroute_duration, degradation_duration = local_path_reroute(G_sample, recovery_parameters, app_path, app_demand, faulty_node, app_access, app_exit)
        #
        # new_subpath_optional = k_shortest_paths(K, G_sample, source, destination,  'weight', app_strategy, original_path_length)
        # new_subpath = find_available_path(G_sample, app_demand, new_subpath_optional, node_fail_list[0]) # 这只能找到子路径上的路径带宽是否满足
        #
        # if new_subpath: # 如果路径存在
        #     redirecting_node = 0
        #     merging_node = original_path_length - fail_node_index # 从合并原路径与恢复路径的节点到target节点的跳数
        #     if fail_node_index == 0 or fail_node_index == len(app_path)-1:
        #         new_app_path = new_subpath
        #         merging_node = 0 #此时选择了新的od对来进行路由，即表示原路径与新路径没有重叠
        #     else:
        #         new_app_path = app_path[:fail_node_index-1] + new_subpath[:-1] + app_path[fail_node_index+1:] # 对子路径进行切片组合为新的业务路径
        #     reroute_duration, degradation_duration = service_recovery_duration(app_strategy, rerouting_app_num, redirecting_node, merging_node, len(new_subpath), path_calculating_time, message_processing_time, time_propagation)
        #     print('成功重路由恢复,源路径为{},新路径为{} '.format(app_path, new_app_path))
        # else:
        #     print('业务重路由不成功,子路径为{}'.format(new_subpath))


    return new_app_path, reroute_duration, degradation_duration


def local_path_reroute(G, recovery_parameters, app_path, app_demand, faulty_node, app_access_lst, app_exit_lst):
    # 尽量利用原有路径的恢复策略
    Max_hop = 15
    K = 5
    message_processing_time, path_calculating_time, rerouting_app_num = recovery_parameters[0], recovery_parameters[1], recovery_parameters[2]
    time_propagation = 0.005
    # Create a list of source-destination pairs based on the service requests
    service_request = {}  # 存储服务可能的请求od
    available_path = []
    new_app_path = None
    reroute_duration = 0
    degradation_duration = 0
    app_strategy = 'Local'
    Flag = False # 标志找到合适路径的标记

    faulty_node_index = app_path.index(faulty_node)  # 找到故障节点在app原路径中的索引
    # set_node_weight(G, app_path, 'faulty')

    if faulty_node_index == 0:
        service_request['source'] = [element for element in app_access_lst if element != faulty_node and element not in app_path]
        service_request['destination'] = app_path[1:]  # 除掉首节点的剩余节点

        redirecting_node = 0  # 重定向的发起节点距离为0
        # 进行重路由计算
        for source in service_request['source']:
            for destination in service_request['destination']:
                new_subpath_optional = k_shortest_paths(K, G, source, destination, 'weight', app_strategy, len(app_path) )

                new_subpath = find_available_path(G, app_path, app_demand, new_subpath_optional, faulty_node)  # 这只能找到子路径上的路径带宽是否满足
                if new_subpath:  # 如果找到了新的子路径
                    merging_node = len(app_path) - app_path.index(new_subpath[-1])
                    repeated_nodes = set(new_subpath)&set(app_path[app_path.index(new_subpath[-1]) + 1:])
                    if repeated_nodes:
                        print('存在重复节点')
                    if (len(new_subpath) + merging_node) < Max_hop :  # 仅当新的路径满足跳数约束才算作重路由成功
                        new_app_path = new_subpath + app_path[app_path.index(new_subpath[-1]) + 1:]
                        reroute_duration, degradation_duration = service_recovery_duration(app_strategy, rerouting_app_num, redirecting_node, merging_node, len(new_subpath), path_calculating_time, message_processing_time, time_propagation)
                        Flag = True
                        break
            if Flag:
                break

    elif faulty_node_index == len(app_path) - 1:
        service_request['source'] = app_path[:-1] # 除掉尾节点的剩余节点
        service_request['source'].reverse() # 将源节点list翻转
        service_request['destination'] = [element for element in app_exit_lst if element != faulty_node and element not in app_path]

        merging_node = 0 # 与原路径重叠的合并跳数为0
        # 进行重路由计算
        for destination in service_request['destination']:
            # set_node_weight(G, [destination], 'alive')
            for source in service_request['source']:  # 该策略下优先调整源节点的选择
                # set_node_weight(G, [source], 'alive')
                new_subpath_optional = k_shortest_paths(K, G, source, destination, 'weight', app_strategy, len(app_path))

                new_subpath = find_available_path(G, app_path, app_demand, new_subpath_optional,  faulty_node)  # 这只能找到子路径上的路径带宽是否满足
                if new_subpath:  # 如果找到了新的子路径
                    redirecting_node = len(app_path) - app_path.index(new_subpath[0])
                    repeated_nodes = set(new_subpath)&set(app_path[: app_path.index(new_subpath[0])]) # 找出子路径是否与原路径存在重复元素
                    if repeated_nodes:
                        print('存在重复节点')
                    if (len(new_subpath) + redirecting_node) < Max_hop:  # 仅当新的路径满足跳数约束才算作重路由成功
                        new_app_path = app_path[: app_path.index(new_subpath[0])] + new_subpath
                        reroute_duration, degradation_duration = service_recovery_duration(app_strategy, rerouting_app_num, redirecting_node, merging_node, len(new_subpath), path_calculating_time, message_processing_time, time_propagation)
                        Flag = True
                        break
            if Flag:
                break

    else:
        service_request['source'] = app_path[:faulty_node_index]
        service_request['source'].reverse()
        service_request['destination'] = app_path[faulty_node_index + 1:]
        # 进行重路由计算
        for source in service_request['source']:
            # set_node_weight(G, [source], 'alive')
            for destination in service_request['destination']:
                # set_node_weight(G, [destination], 'alive')
                new_subpath_optional = k_shortest_paths(K, G, source, destination, 'weight', app_strategy, len(app_path))
                new_subpath = find_available_path(G, app_path, app_demand, new_subpath_optional, faulty_node)  # 这只能找到子路径上的路径带宽是否满足
                if new_subpath:  # 如果找到了新的子路径
                    merging_node = len(app_path) - app_path.index(new_subpath[-1])
                    redirecting_node = app_path.index(new_subpath[0])

                    repeated_nodes = set(new_subpath)&set(app_path[:app_path.index(new_subpath[0])] + app_path[app_path.index(new_subpath[-1]) + 1:]) # 找出子路径是否与原路径存在重复元素
                    if repeated_nodes:
                        print('存在重复节点')
                    if (len(new_subpath) + merging_node + redirecting_node) < Max_hop:  # 仅当新的路径满足跳数约束才算作重路由成功
                        new_app_path = app_path[:app_path.index(new_subpath[0])] + new_subpath + app_path[app_path.index(new_subpath[-1]) + 1:]
                        reroute_duration, degradation_duration = service_recovery_duration(app_strategy, rerouting_app_num, redirecting_node, merging_node, len(new_subpath), path_calculating_time, message_processing_time, time_propagation)
                        Flag = True
                        break
            if Flag:
                break

    return new_app_path, reroute_duration, degradation_duration

def set_node_weight(G, node_lst, setting_state):
    # 将给定网络节点集合中的各节点的相邻链路权重置为对应的状态
    for n in node_lst:
        if setting_state == 'faulty':
            adj_nodes = list(G.adj[n])
            for adj in adj_nodes:
                G.adj[n][adj]['weight'] = float('inf')  # 将节点邻接的边的权重设置为无穷大
        else:
            adj_nodes = list(G.adj[n])
            for adj in adj_nodes:
                G.adj[n][adj]['weight'] = 1 - G.adj[n][adj]['fail_rate'] # 将节点邻接的边的权重设置为无穷大

def find_available_path(G, app_original_path, app_demand, path_list, faulty_node):
    # First-fit策略：从K最短路集合中找出第一条满足业务带宽需求的路径
    # 如果所有候选路径均不满足业务的带宽需求，则随机选择一条路径作为available＿path
    available_path = None
    app_residual_path = []
    faulty_node_index = app_original_path.index(faulty_node)  # 找到故障节点在app原路径中的索引


    for shortest_path in path_list:
        # Check if the path has enough available bandwidth (first-fit strategy)
        if faulty_node_index == 0:
            destination = app_original_path.index(shortest_path[-1])
            app_residual_path = app_original_path[destination + 1:]
        elif faulty_node_index == len(app_original_path)-1:
            source = app_original_path.index(shortest_path[0])
            app_residual_path = app_original_path[:source]
        else:
            source = app_original_path.index(shortest_path[0])
            destination = app_original_path.index(shortest_path[-1])
            app_residual_path = app_original_path[:source] + app_original_path[destination+1:]
        # app_available_path = [node for node in app_original_path if node != source or node != destination] # 除掉源宿节点以外的业务原路径上的节点
        repeated_nodes = set(shortest_path) & set(app_residual_path)
        if faulty_node in shortest_path or repeated_nodes: # 如果计算出来的路径包含原路径上的节点，则表示存在回环
            continue
        else:
            available_path_bandwidth = min(G[u][v]['capacity'] - G[u][v]['load'] for u, v in zip(shortest_path, shortest_path[1:]))
            path_weight  = sum(G[u][v]['weight'] for u, v in zip(shortest_path, shortest_path[1:]))

            if available_path_bandwidth >= app_demand and path_weight < 100:
                available_path = shortest_path
                break

    return available_path


def service_recovery_duration(app_strategy, current_service_num, redirecting_node, merging_node, new_path_length,time_path_cal, time_process, time_propagation):
    # 计算服务在恢复过程中的时长
    disruption_time = 0 # 服务重路由计算的时长
    degradation_time = 0 # 服务故障定位以及新路径响应的时长
    strategy = app_strategy # 服务的重路由策略

    disruption_time = current_service_num*time_path_cal + 2 * new_path_length * time_propagation
    if strategy == 'Global':
        degradation_time = (current_service_num + redirecting_node) * time_process + (redirecting_node+new_path_length) * time_propagation
    else:
        degradation_time = current_service_num*time_process + (merging_node+new_path_length) * time_propagation

    return disruption_time, degradation_time



if __name__ == '__main__':
    # 调试尽量分离的重路由算法；
    import_file = False
    Node_num =  100
    Topology = 'Random'
    Area_size = (250, 150)
    Area_width, Area_length = 250, 150
    Coordinates = generate_positions(Node_num, Area_width, Area_length, False)

    # TX_range = 50 # 传输范围为区域面积的1/5时能够保证网络全联通
    transmit_power = 15  # 发射功率(毫瓦)，统一单位：W
    path_loss = 2.5  # 单位：无
    noise = pow(10, -10)  # 噪声的功率谱密度(毫瓦/赫兹)，统一单位：W/Hz, 参考自https://dsp.stackexchange.com/questions/13127/snr-calculation-with-noise-spectral-density
    bandwidth = 20 * pow(10, 6)  # 带宽(Mhz)，统一单位：Hz
    lambda_TH = 8 * pow(10, -1)  # 接收器的敏感性阈值,用于确定节点的传输范围
    TX_range = pow((transmit_power / (bandwidth * noise * lambda_TH)), 1 / path_loss)
    CV_range = 30 # 节点的覆盖范围

    # 业务请求的参数
    App_num = 20
    grid_size = 5
    traffic_th = 0.5 # 业务网格的流量阈值
    App_Demand = np.random.normal(loc= 5, scale=1, size=App_num) # 生成平均值为5，标准差为1的业务带宽请求的整体分布
    App_Priority = [1,2,3]
    ratio_str = 0 # 尽量分离和尽量重用的业务占比
    Strategy_P = ['Global'] * int(App_num*(1-ratio_str))
    Strategy_S = ['Local'] * int(App_num*ratio_str)
    App_Strategy = Strategy_S + Strategy_P

    G = Network(Topology, Node_num, Coordinates, TX_range, transmit_power, bandwidth, path_loss, noise, import_file)
    G, Apps = init_func(G, Coordinates, Area_size, CV_range,  grid_size, traffic_th, App_num, App_Demand, App_Priority, App_Strategy)

    fail_node = [Apps[1].path[-2] ]
    component_failure(G, fail_node)
    new_app_path = path_reroute(G, Apps[1].access, Apps[1].exit, Apps[1].path, Apps[1].str, fail_node)

    new_app_load = app_load_allocating(G, Apps, 1, new_app_path)

    print(new_app_load)


