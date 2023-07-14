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
    new_app_path = [] # 业务重路由的候选路径集合
    new_app_load = 0
    reroute_duration = 0
    source = app_path[0]
    destination = app_path[-1]

    '''业务的重路由分为2种策略：Global和Local'''
    if app_strategy == 'Global':
        # 全局重路由策略(从源宿节点重新找一条路)
        # print('业务的重路由策略为{}'.format(app_strategy))

        # 1) 先将原始的路径上的链路权重设置为无穷大
        # for i in range(len(app_path)-1):
        #     G_sample.adj[app_path[i]][app_path[i+1]]['weight'] = float('inf')
        # 2) 然后在业务的接入和接出节点集合中寻找一条满足业务带宽需求的路径
        node = node_fail_list[0] # 取出故障节点集合中的第一个节点
        fail_node_index = app_path.index(node)
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
                    return new_app_path, reroute_duration
            else:
                return new_app_path, reroute_duration

        elif fail_node_index == len(app_path) - 1:
            if len(app_exit) > 1:
                available_exit_list = []
                for n in app_exit:
                    if G_sample.nodes[n]['alive'] == 1:
                        available_exit_list.append(n)
                if available_exit_list:  # 如果存在可接入的节点list
                    destination = random.choice(available_exit_list)
                else:  # 如果业务仅1个接入节点且发生了故障，则直接返回空的路径
                    return new_app_path, reroute_duration

            else:
                return new_app_path, reroute_duration

        else: # 如果故障的节点为业务的中继节点，则直接从其源宿节点list中随机选择一个进行重路由
            source = app_path[0]
            destination = app_path[-1]
        # 计算新的业务路径
        # new_app_path = nx.shortest_path(G_sample, source, destination, 'weight')
        original_path_length = len(app_path)
        new_app_path_optional = k_shortest_paths(K, G_sample, source, destination,  'weight', app_strategy, original_path_length)
        # print('计算得到的可选路径为{}'.format(new_app_path_optional))
        new_app_path = find_available_path(G_sample, app_demand, new_app_path_optional, node_fail_list[0])
        if new_app_path: # 如果路径存在
            reroute_duration = (rerouting_app_num + (fail_node_index + 1) + 2 * len(new_app_path)) * message_processing_time + rerouting_app_num * path_calculating_time
        else:
            print('业务重路由计算不成功, 原路径长度为{}，可选的路径集合为{}'.format(original_path_length, new_app_path_optional))

    '''业务的Local重路由策略'''
    if app_strategy == 'Local':
    # 找出节点故障导致的业务原路径中故障的链路
    #     print('业务的重路由策略为{}'.format(app_strategy))
        # 先将原路径上链路权重设置为inf，避免计算到的最短路径出现环路
        for i in range(len(app_path) - 1):
            G_sample.adj[app_path[i]][app_path[i + 1]]['weight'] = float('inf')

        node = node_fail_list[0]
        fail_node_index = app_path.index(node)
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
                    return new_app_path, reroute_duration
            else:
                return new_app_path, reroute_duration

        elif fail_node_index == len(app_path)-1:
            # fail_link = (app_path[fail_node_index-1], node)
            # link_fail_list.append(fail_link)
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
                    return new_app_path, reroute_duration
            else: # 如果业务的接出节点仅有1个
                return new_app_path, reroute_duration
        else: # 如果业务的故障模式为中继节点
            source = app_path[fail_node_index - 1]
            destination = app_path[fail_node_index + 1]

        # 在计算子路径之前，将现有的可用节点的相邻链路的权重设置为无穷
        for n in app_path:
            if n != source and n!= destination: # G_sample.nodes[n]['alive'] == 1
                # 将除了源宿节点外的其他节点的相邻链路都设置为inf,避免路由计算中加入这些节点(还需要加入故障节点,因为主循环中对链路设置为inf可能没有传进来)
                adj_nodes = list(G_sample.adj[n])
                for adj in adj_nodes:
                    G_sample.adj[n][adj]['weight'] = float('inf')  # 将节点邻接的边的权重设置为无穷大

        # new_subpath = nx.shortest_path(G_sample, source, destination, 'weight')

        original_path_length = len(app_path)
        new_subpath_optional = k_shortest_paths(K, G_sample, source, destination,  'weight', app_strategy, original_path_length)

        # print('重路由计算得到的候选路径集为{}'.format(new_subpath_optional))
        new_subpath = find_available_path(G_sample, app_demand, new_subpath_optional, node_fail_list[0]) # 这只能找到子路径上的路径带宽是否满足
        # print('最终选择的新路径为{} '.format(new_subpath))
        if new_subpath: # 如果路径存在
            # if fail_node_index == 0 : # 如果故障节点为首/尾节点,则不对计算得到的新路径进行切片
            #     new_app_path = new_subpath[:-1] + app_path[fail_node_index+1:]
            # elif fail_node_index == len(app_path)-1:
            #     new_app_path = app_path[:fail_node_index-1] + new_subpath
            if fail_node_index ==0 or fail_node_index == len(app_path)-1:
                new_app_path = new_subpath
            else:
                new_app_path = app_path[:fail_node_index-1] + new_subpath[:-1] + app_path[fail_node_index+1:] # 对子路径进行切片组合为新的业务路径
            reroute_duration = (rerouting_app_num + 2 * len(new_subpath)) * message_processing_time + rerouting_app_num * path_calculating_time
            # print('成功重路由恢复,源路径为{},新路径为{} '.format(app_path, new_app_path))
        else:
            print('业务重路由计算不成功，原路径长度为{}，可选的路径集合为{}'.format(original_path_length, new_subpath_optional))

        # # 用来检测Local策略下重路由计算出来的路径是否有回环
        # s1 = set(new_app_path)
        # for n in s1:
        #     if new_app_path.count(n) > 1:
        #         print("元素{},重复{}次".format(n, new_app_path.count(n))) # 统计业务路径中出现重复的节点

    return new_app_path, reroute_duration

def load_allocate(G, evo_time, app_new_path, app_demand, app_original_load, app_downtime):
    # 根据新生成的app_path_list来计算最终分配的业务负载
    app_degradation = {} # 记录业务降级的负载以及降级的持续时间
    bottleneck_link_list =[] # 找出所有瓶颈链路的集合
    app_allocated_load = app_demand
    app_degradation_time = 0 # 记录业务降级发生的时刻
    # 根据重路由成功的业务的路径,找出所有过载的链路

    for i in range(len(app_new_path)-1):
        link_available_cap = G.adj[app_new_path[i]][app_new_path[i+1]]['capacity'] - G.adj[app_new_path[i]][app_new_path[i+1]]['load']
        # print('链路剩余可用容量为{}'.format(link_available_cap))
        if link_available_cap < app_demand:
            if link_available_cap < 0.1:
                print('当前链路的剩余可用容量为{}'.format(link_available_cap))
                print('app_new path is {}'.format(app_new_path))
                print('瓶颈链路为{}\n'.format([app_new_path[i],app_new_path[i+1]]))
            bottleneck_link_list.append((app_new_path[i],app_new_path[i+1]))
            # 若当前业务分配的负载大于链路的剩余容量，则将业务负载更新为链路剩余容量
            if  app_allocated_load > link_available_cap :
                app_allocated_load = link_available_cap
    # 计算业务的降级时长
    if app_allocated_load < app_demand: # 如果当前时刻分配的负载小于业务需求，则发生了降级
        app_degradation_time = evo_time # 更新业务发生降级的时间
        # 进一步判断是初次降级还是连续降级
        if app_downtime > 0: # 判断上一时刻业务是否降级
            degradation_load = app_original_load # 记录上一时刻的业务负载为降级的负载
            app_degradation[degradation_load] = evo_time - app_downtime
            # print('业务的降级时长为{}'.format(app_degradation))
            # if evo_time - app_downtime > 100:
            #     print('当前演化时刻为{},业务上一降级的时刻为{}'.format(evo_time, app_downtime))
    else:
        app_degradation_time = 0

    return app_allocated_load, app_degradation, app_degradation_time

def app_degradation(evo_time, app_demand, app_original_load, app_allocated_load, app_downtime):
    app_degradation = {} # 业务降级时的负载: 降级的持续时间
    app_degradation_time = 0 # 业务发生降级的时刻
    if app_allocated_load < app_demand: # 如果当前时刻分配的负载小于业务需求，则发生了降级
        app_degradation_time = evo_time # 更新业务发生降级的时间
        # 进一步判断是初次降级还是连续降级
        if app_downtime > 0: # 判断上一时刻业务是否降级
            degradation_load = app_original_load
            # if evo_time == app_downtime:
            #     print('业务当前的降级不被记录')
            app_degradation[degradation_load] = evo_time - app_downtime
    else:
        app_degradation_time = 0

    return app_degradation, app_degradation_time

def find_available_path(G, app_demand, path_list, fail_node):
    # First-fit策略：从K最短路集合中找出第一条满足业务带宽需求的路径
    # 如果所有候选路径均不满足业务的带宽需求，则随机选择一条路径作为available＿path
    available_path = []
    available_load = 0
    # print('业务的候选路径集合为{},故障节点为{}'.format(path_list, fail_node))
    removed_path = []
    for path in path_list:
        if fail_node in path: # 如果故障节点在新计算出来的路径中，则直接跳过该路径
            removed_path.append(path)
            continue
        else:
            flag = len(path)-1
            index = 0
            link_load_path = []
            for i in range(len(path) - 1):
                link_available_cap = G.adj[path[i]][path[i + 1]]['capacity'] - G.adj[path[i]][path[i + 1]]['load']
                link_load_path.append(link_available_cap)
                weight = G.adj[path[i]][path[i + 1]]['weight'] # 排除掉那些最短路计算出来为中断的链路
                if link_available_cap < 0.05 or weight == float('inf'): # 如果路径剩余可用带宽小于0，则跳出循环寻找下一条路径
                    # 这里主要的问题是 有的链路上的负载超过了容量,有的链路的权重为inf也被计算进来了
                    # print('路径{}计算错误,链路{}的权重为{},链路的剩余可用带宽为{}'.format(path, (path[i],path[i+1]),weight, link_available_cap))
                    removed_path.append(path) # 在路径集合中移除该错误路径,最后再对移除的path_list进行操作
                    break # 跳出当前循环
                else:
                    index += 1

            if index == flag and min(link_load_path) > app_demand: # 如果索引到该path中的最后一条链路仍然有可用的带宽的话，则终止循环，输出该路径为业务的路径
                available_path = path  # 找到可用路径后跳出循环
                break
                # print('计算得到的可用的路径为{}'.format( available_path ))
    # print('待移除的业务路径为{}'.format(removed_path))

    for l in removed_path:
        path_list.remove(l)# 将计算错误的路径移除出去

    if not available_path and path_list: # 如果业务的可用带宽仍然为空(即所有的子路径均不满足业务的带宽需求)
        available_path = random.choice(path_list)
    #     print('当前剩余可用的路径集合为{}'.format(path_list))
    #     print('随机选择的业务路径为{}'.format(available_path))
    #
    # print('业务选择的路径为{}'.format(available_path))

    return available_path




# 资源（带宽）分配规则
def app_load_allocating(G, Apps, app_id, app_new_path):
    # 根据业务的带宽需求和部署路径, 来确定业务在网络各链路上的负载(K is the scaling factor)
    app_demand = Apps[app_id].demand
    # 1. 先计算业务新的路径上链路的最小可用容量
    link_load_path = [] # 记录业务候选路径上的链路剩余可用容量
    bottleneck_link = []
    App_allocated_load = {} # 记录发生负载更新的业务id及其新负载
    app_allocated_load = app_demand # 初始分配带宽为业务需求

    for i in range(len(app_new_path)-1):
        # print('current link is {}'.format(G.edges[new_app_path[i], new_app_path[i+1]]))
        link_available_cap = G.adj[app_new_path[i]][app_new_path[i+1]]['capacity'] - G.adj[app_new_path[i]][app_new_path[i+1]]['load']
        link_load_path.append(link_available_cap)
        if link_available_cap < app_allocated_load: # 如果当前链路可用容量小于业务带宽则记录为“瓶颈链路”
            bottleneck_link.append((app_new_path[i], app_new_path[i+1]))
            app_allocated_load = link_available_cap # 将剩余的链路容量分配给业务

    capacity_available = min(link_load_path) # 计算业务路径上的最小带宽
    # if app_allocated_load < app_demand:
    #     print('瓶颈链路为{},分配的业务带宽为{}'.format(bottleneck_link, app_allocated_load))
    #
    # return app_allocated_load
    # 2. 判断当前路径上的剩余可用容量是否满足业务的带宽需求
    if capacity_available >=  app_demand:
        App_allocated_load[app_id] = app_demand
    else:
        # 如果剩余可用容量不足，则根据瓶颈链路上部署的业务来进行缩放
        print('瓶颈链路为{}'.format(bottleneck_link))

        for e in bottleneck_link: # 依次遍历各瓶颈链路上的业务，对各业务的分配负载进行缩放
            link_capacity = G.adj[e[0]][e[1]]['capacity']
            app_list = G.adj[e[0]][e[1]]['app_dp'] + [app_id] # 链路上部署的业务集合应该加上此时待部署的业务id
            app_original_load = [Apps[id].load for id in app_list]
            app_pri = [Apps[id].SLA for id in app_list]  # 读取链路上部署的业务的SLA
            total_pri = sum(app_pri)
            while True:
                # 计算各业务负载的缩放系数
                app_scaling = [pri / total_pri for pri in app_pri]
                app_scaling_factor = [(1 - k) / (1 + k) for k in app_scaling]
                app_new_load = [x * y for x, y in zip(app_original_load, app_scaling_factor)]  # 计算调整后的各业务负载
                if sum(app_new_load) <= link_capacity:  # 如果调整后业务的负载满足链路的容量约束, 则更新业务子路径上的负载
                    print('当前计算得到的链路{}上的业务负载分配为{}'.format(e, app_new_load ))
                    # G.adj[e[0]][e[1]]['load'] = sum(app_new_load)  # 更新链路的负载为缩放业务带宽后的负载(这一步骤转移至主函数的app_deploy_edge中)
                    for i in range(len(app_list)):
                        # 比较当前的业务负载和原有的业务负载的值的大小，如果小于，则更新App中对应子路径的业务负载值,并同时更新对应链路的负载值
                        # if app_new_load[i] < app_original_load[i]:
                        if app_list[i] in App_allocated_load: # 如果某一业务的多条链路上的负载进行了更新，则取最小的那个分配负载作为业务的load
                            current_app_load = App_allocated_load[app_list[i]]
                            if current_app_load > app_new_load[i]: # 仅当当前链路上分配给业务的load大于之前所分配的load，才对业务的最终分配load进行更新
                                App_allocated_load[app_list[i]] = app_new_load[i]
                                print('分配给业务{}的原负载为{}'.format( app_list[i], current_app_load))
                                print('分配给业务{}的新负载为{}'.format(app_list[i], app_new_load[i]))
                        else:
                            App_allocated_load[app_list[i]] = app_new_load[i]
                            print('重新分配负载的业务集合为{}'.format(App_allocated_load))
                    break
                else:
                    app_original_load = app_new_load


    return App_allocated_load


if __name__ == '__main__':
    # 调试尽量分离的重路由算法；
    import_file = False
    Node_num =  100
    Topology = 'Random'
    Area_size = (250, 150)
    Area_width, Area_length = 250, 150
    Coordinates = generate_positions(Node_num, Area_width, Area_length)

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


