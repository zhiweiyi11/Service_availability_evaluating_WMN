#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   ：Service_availability -> FunctionTest
@IDE    ：PyCharm
@Author ：Yi Zhiwei
@Date   ：2023/3/12 17:30
@Desc   ：用于测试一些简单的函数功能
=================================================='''
import os
import threading
import time
from multiprocessing import cpu_count, Pool
import math

import networkx as nx
import numpy as np
import random
from concurrent.futures import ThreadPoolExecutor
from sympy import *
import pandas as pd
from sympy import exp

from Evaluating_Scripts.Calculating_Availability import calculateAvailability
from Evolution_Model.Evolution_Rules import *

def RecursionFunc(arr1,arrList):
    # 从arrList中各取出一个元素，并进行组合
    if (arrList):
        string = []
        for x in arr1:
            for y in arrList[0]:
                string.append((x, y))
        result = RecursionFunc(string,arrList[1:])
        return result
    else:
        return arr1

def route_search(G, source, destination, app_demand, routing_th):
    reroute_times = 0
    path_exist = 1
    new_app_path = 0
    path = nx.shortest_path(G, source, destination, 'weight')
    # 计算路径的存在概率
    for i in range(len(path) - 1):
        delta = random.random()
        link_fail_rate = G.adj[path[i]][path[i + 1]]['fail_rate']
        cap_available = min( G.nodes[path[i]]['cap'], G.nodes[path[i+1]]['cap'])
        # print('生成的随机数为{}'.format(delta))
        # print('链路的故障率为{}\n'.format(link_fail_rate))

        if delta > link_fail_rate and cap_available >= app_demand:  # 如果随机数大于该链路的中断率，则该链路不存在，需要重新选路
            new_app_path = path
            break
        else:
            reroute_times += 1
            G.adj[path[i]][path[i + 1]]['weight'] = float('inf') # 置当前链路的权重为无穷大
            if reroute_times > routing_th:
                # 如果重路由次数超过了业务重路由的阈值
                new_app_path = []
                break

    return new_app_path, reroute_times


def async_add(max):
    sum = 0
    for i in range(max):
        sum += i
    # time.sleep(1)
    # print(threading.current_thread().name + '执行求和操作的结果是={} \n'.format(sum) )
    return sum

def find_path(G, source, destination):
    path = nx.shortest_path(G, source, destination)
    print(threading.current_thread().name + '执行操作的路径结果是={} \n'.format(path))
    time.sleep(0.5)
    return path

def multi_threading_func(N,  G):
    # 测试结果，用简单函数进行测试发现多线程并发的计算结果更快
    Nodes = list(G)

    with ThreadPoolExecutor(max_workers=20) as pool:
        for i in range(N):
            source = random.choice(Nodes)
            destination = random.choice(Nodes)
            future = pool.submit(find_path, G, source, destination)
            print(future.done())
        print('------------主线程执行结束--------------')

def multi_threading_test():
    # 多线程模块调用的基本方法
    st1 = time.time()
    Res = []
    with ThreadPoolExecutor(max_workers=20) as pool:
        # 向线程池提交一个task
        for i in range(100):
            future1 = pool.submit(async_add, i)  # 向线程池提交一个task, 20作为async_add()函数的参数

            # future2 = pool.submit(async_add, 50)
            def get_result(future):
                print(threading.current_thread().name + '运行结果：' + str(future.result()) + '\n')
                # 这里存储的结果是乱序的，因为不同的线程执行结束的时间不确定
                Res.append(future.result())

            future1.add_done_callback(get_result)

        # 测试线程是否结束，判断future1代表的任务是否执行完
        print(future1.done())

        # 定义获取结果的函数, 这样可以非阻塞地获取线程执行的结果
        # def get_result(future):
        #
        #     print(threading.current_thread().name + '运行结果：' + str(future.result()) +'\n')
        #     return future.result()

        # 通过add_done_callback函数向线程池中注册了一个获取线程执行结果的函数get_result
        ## 查看future1代表的任务返回的结果
        # future1.add_done_callback(get_result)
        ## 查看future2代表的任务的返回结果
        # future2.add_done_callback(get_result)

        print('------------主线程执行结束------------')
    et1 = time.time()
    print('多线程并发计算的时长为{}s\n'.format(et1 - st1))

    st2 = time.time()
    res = []
    for i in range(100):
        res.append(async_add(i))
    et2 = time.time()
    print('for循环计算的时长为{}s \n'.format(et2 - st2))

def Apps_availability_func(N, args, pool_num):
    '''
    # 计算业务可用度的主函数(调用多进程并行计算)
    :param N:  网络演化的次数
    :param args:  网络演化模型的参数
    :param pool_num:  开启进程池的数量
    :return multi_avail: 各业务可用度的结果
    '''
    multi_avail = pd.DataFrame(index=list(args[2].keys())) # 存储N次演化下各次的业务可用度结果
    multi_loss = pd.DataFrame(index=list(args[2].keys())) # 存储N次演化下各次业务的带宽损失结果

    print('CPU内核数为{}'.format(cpu_count()))
    print('当前母进程为{}'.format(os.getpid()))
    pool = Pool(pool_num)  # 开启pool_num个进程, 需要小于本地CPU的个数

    for n in range(N):
        # t1 = time.time()
        # functions are only picklable if they are defined at the top-level of a module.(函数尽可以被调用当其位于模块中的顶层)
        res = pool.apply_async(func=calculateAvailability, args=args)  # 使用多个进程池异步进行计算，apply_async执行函数,当有一个进程执行完毕后，会添加一个新的进程到pool中
        multi_avail.loc[:, n + 1] = pd.Series(res.get()[0])  # 将单次演化下各业务的可用度结果存储为dataframe中的某一列(index为app_id)，其中n+1表示列的索引
        multi_loss.loc[:, n + 1] = pd.Series(res.get()[1])
        # t2 = time.time()
        # print('------------------分隔线---------------当前完成第{}次演化，耗时{}min'.format(n, (t2-t1)/60))

    pool.close()
    pool.join()  # #调用join之前，一定要先调用close() 函数，否则会出错, close()执行后不会有新的进程加入到pool,join函数等待素有子进程结束

    return multi_avail, multi_loss

def function_SINR(s, d_ij, d_interferer_list, noise_power):
    m = 4
    gamma = 1
    W = 0.0002
    # d_ij = 20
    # d_interferer_list = [35]
    alpha = 2
    Pt = 1.5
    # s = symbols('s', real=True)
    k = (m * gamma * noise_power) / (Pt*pow(d_ij, -alpha))

    y1 = 1
    for d_vj in d_interferer_list:
        b = gamma * pow((d_vj / d_ij), -alpha)
        print('b 的值为{}'.format(b))
        y1 *= pow( (1+s*b), -m)
    y2 = exp(-k*s)  * y1

    return y2


def calculate_SINR_outage_probability(m, d_ij, d_interferer_list, noise_power):
    '''
    # 计算基于SINR的链路故障率
    :param gamma: SINR 阈值, 通常为1
    :param W: 噪声功率, 0.0001 W
    :param alpha: 路损指数, 在[2,6]之间取值
    :param m: 衰落系数的形状参数, 4
    :param d_ij: 链路(v_i, v_j)的距离
    :param d_interferer_list: 干扰节点的距离集合
    :return:
    '''
    res = 0
    s = symbols('s', real= True)
    for n in range(m):
        aa = pow(-1, n)/np.math.factorial(n)
        print('aa is {}'.format(aa))
        bb = diff(function_SINR(s, d_ij, d_interferer_list, noise_power), s , n).subs({s:1})
        print('bb is {}'.format(bb))
        res += aa*bb
    outage_probability = 1- res
    return outage_probability

def degradation_time_calculate():
    # 存放计算多业务降级时长的代码
    pass
    '''# action4: 对业务进行带宽分配操作
    app_original_load = app.load
    app_new_load, degradation_duration, degradation_time = load_allocate(G_tmp, evo_time, app.path, app.demand, app_original_load, app.down_time)
    app.down_time = degradation_time
    app.load = app_new_load
    if degradation_duration:  # 如果存在降级，则将降级时长及负载记录
        app.outage['degradation'].append(degradation_duration)
    app.app_deploy_edge(G_tmp)
    '''

    ''' # 这里为对多业务的带宽分配
    # action4 :对多业务进行带宽分配的调整
    App_allocated_load = app_load_allocating(G_tmp, App_tmp, app_id, app_new_path)
    # print('业务带宽重分配的结果为{}'.format(App_allocated_load))
    for id, load in App_allocated_load.items(): # 把重路由部署的业务放到最后一个更新,
        if id == app_id:
            continue
            # app_original_load = App_tmp[id].load # 记录业务的原负载
            # App_tmp[id].load = load # 赋值给业务的负载
            # App_tmp[id].app_deploy_edge(G_tmp) # 将业务的负载部署至网络上
            # # print('业务{}的新路径{}被重新部署至网络上 \n'.format(id, App_tmp[id].path))
            # degradation_duration, degradation_time = app_degradation(evo_time, App_tmp[id].demand, app_original_load, load, App_tmp[id].down_time)
            # if App_tmp[id].down_time == evo_time and App_tmp[id].outage['degradation']: # 如果降级发生在当前的演化时刻且业务之前已经发生过降级
            #     if degradation_duration:  # 如果存在降级，则将降级时长及负载记录
            #         duration = list(App_tmp[id].outage['degradation'][-1].values())[0] # 读取上一次记录的业务降级的持续时间
            #         # print('业务{}的旧负载为{}'.format(id, app_original_load))
            #         App_tmp[id].outage['degradation'][-1] = {load: duration} # 仅替换降级发生时的负载为当前业务分配的负载load
            #         # print('业务{}降级时负载替换成功为{}'.format(id, {load:duration}))
            # else:
            #     if degradation_duration:  # 如果存在降级，则将降级时长及负载记录
            #         App_tmp[id].outage['degradation'].append(degradation_duration)
            # App_tmp[id].down_time = degradation_time # 更新业务降级的时刻
        else:
            # print('app {} 属于被动重分配'.format(id))
            app_original_load = App_tmp[id].load # 记录业务的原负载
            App_tmp[id].app_undeploy_edge(G_tmp) # 先将业务原来映射路径上各链路的负载更新
            # print('业务{}解除到链路映射时的路径为{}'.format(id, App_tmp[id].path))
            App_tmp[id].load = load # 赋值给业务的负载
            App_tmp[id].app_deploy_edge(G_tmp)
            print('业务{}恢复到链路映射时的路径为{}'.format(id, App_tmp[id].path))
            degradation_duration, degradation_time = app_degradation(evo_time, App_tmp[id].demand, app_original_load, load, App_tmp[id].down_time)

            if App_tmp[id].down_time == evo_time and App_tmp[id].outage['degradation']: # 如果降级发生在当前的演化时刻且业务之前已经发生过降级
                if degradation_duration:  # 如果存在降级，则将降级时长及负载记录
                    duration = list(App_tmp[id].outage['degradation'][-1].values())[0] # 读取上一次记录的业务降级的持续时间
                    # print('业务{}的旧负载为{}'.format(id, app_original_load))
                    App_tmp[id].outage['degradation'][-1] = {load: duration} # 仅替换降级发生时的负载为当前业务分配的负载load
                    # print('业务{}降级时负载替换成功为{}'.format(id, {load:duration}))
            elif App_tmp[id].down_time == evo_time and not App_tmp[id].outage['degradation']: # 如果业务是首次降级，并且在该次演化下发生多次降级
                # 这种情况下只要更新业务的load就行,因为业务才刚刚开始降级
                continue
            else:
                if degradation_duration:  # 如果存在降级，则将降级时长及负载记录
                    App_tmp[id].outage['degradation'].append(degradation_duration)
            App_tmp[id].down_time = degradation_time # 更新业务降级的时刻

    # 最后更新重路由成功的业务的负载
    app_original_load = App_tmp[app_id].load  # 记录业务的原负载
    App_tmp[app_id].load = App_allocated_load[app_id]  # 赋值给业务的负载
    App_tmp[app_id].app_deploy_edge(G_tmp)  # 将业务的负载部署至网络上
    # print('业务{}的新路径{}被重新部署至网络上 \n'.format(id, App_tmp[id].path))
    degradation_duration, degradation_time = app_degradation(evo_time, App_tmp[app_id].demand,
                                                             app_original_load, App_tmp[app_id].load, App_tmp[app_id].down_time)
    if App_tmp[app_id].down_time == evo_time and App_tmp[app_id].outage['degradation']:  # 如果降级发生在当前的演化时刻且业务之前已经发生过降级
        if degradation_duration:  # 如果存在降级，则将降级时长及负载记录
            duration = list(App_tmp[app_id].outage['degradation'][-1].values())[0]  # 读取上一次记录的业务降级的持续时间
            # print('业务{}的旧负载为{}'.format(id, app_original_load))
            App_tmp[app_id].outage['degradation'][-1] = {App_tmp[app_id].load: duration}  # 仅替换降级发生时的负载为当前业务分配的负载load
            # print('业务{}降级时负载替换成功为{}'.format(id, {load:duration}))
    else:
        if degradation_duration:  # 如果存在降级，则将降级时长及负载记录
            App_tmp[app_id].outage['degradation'].append(degradation_duration)
    App_tmp[app_id].down_time = degradation_time  # 更新业务降级的时刻
    '''
def find_alternate_route(G, original_route, faulty_node, app_demand, access_list, exit_list):
    # 沿着原路径，逐跳寻找符合要求的最短路径【尽量利用原路径策略】
    alternate_route = []
    # Create a copy of the original graph
    alternate_graph = G.copy()
    # Remove the faulty node from the alternate graph
    alternate_graph.remove_node(faulty_node)
    # 确定故障节点在原路径中的位置
    faulty_index = original_route.index(faulty_node)

    # Initialize the alternate route list with the original route
    base_route = original_route.copy()
    base_route.remove(faulty_node) #移除掉原路由中的故障节点的route
    if faulty_index == 0:
        access_list.remove(faulty_node)
        if len(access_list) > 0:  # 如果业务有其余可以接入的节点list
            available_access_list = []  # 当前演化态下存储可接入节点的list
            for n in access_list:
                if  n not in original_route: # 避免从已有的路径中选择source节点
                    available_access_list.append(n)
            if available_access_list:  # 如果存在可接入的节点list
                new_source = random.choice(available_access_list)
                base_route.insert(0, new_source) # 在list的头部添加新的路由节点
            else:
                return alternate_route
    if faulty_index == len(original_route)-1:
        exit_list.remove(faulty_node)
        if len(exit_list) > 0:
            available_exit_list = []
            for n in exit_list:
                if n not in original_route:
                    available_exit_list.append(n)
            if available_exit_list:
                new_destination = random.choice(available_exit_list)
                base_route.append(new_destination)
            else:
                return alternate_route
    left = 0
    right = 1
    # 采用双指针来遍历求解Local策略的尽量重用最短路径
    alternate_route.append(base_route[0])

    # 1.直接计算绕过 faulty_node 的恢复路径

    # 2. 如果恢复路径长度超过5跳，则向前继续计算；

    # 3. 直到遍历完整个路径或者是找到符合要求的路径跳出循环

    for i in range(1, min(5, len(original_route) - faulty_index)):
        u = original_route[faulty_index-1]
        v = original_route[faulty_index+1]
        res_paths = list(islice(nx.shortest_simple_paths(alternate_graph, u, v, 'weight'), 5))

    while right < len(base_route): #循环结束的条件为右边指针到达base_route的末尾位置
        u = base_route[left]
        v = base_route[right]
        if alternate_graph.has_edge(u, v):
            # If the edge is present, add it to the alternate route
            alternate_route.append(v)
            alternate_graph.remove_node(u) # 删除已确定为业务路由的节点
            left += 1
            right += 1
        else:
            try:
                # shortest_path = nx.shortest_path(alternate_graph, u, v, 'weight')
                res_paths = list(islice(nx.shortest_simple_paths(alternate_graph, u, v, 'weight'), 5))
                available_path = []
                for path in res_paths:
                    weight_sum = sum(alternate_graph[i][j]['weight'] for i, j in zip(path, path[1:]))
                    if weight_sum != float('inf'):
                        available_path.append(path)

                shortest_path_list = []
                # find_available_path(G, app_demand, shortest_path, faulty_node)
                for p in available_path: # 从可用路径集中找到跳数不超过5的路径
                    if len(p) < 5:  # 如果最短路径的跳数超过5跳则重新选择最短路计算的源节点
                        dup = find_duplicates(p)
                        if dup:
                            print('当前计算的最短路径存在重复节点\n')
                        shortest_path_list.append(p)
                shortest_path = find_available_path(G, app_demand, shortest_path_list, faulty_node)
                if shortest_path: # 如果找到了符合条件的最短路
                    alternate_route += shortest_path[1:]
                    left += 1
                    right += 1
                else:
                    right += 1

            except nx.NetworkXNoPath:
                return None
    # 最后校验一下是否找到符合源宿节点要求的路径
    destination = alternate_route[-1]
    if destination != base_route[-1]:
        alternate_route = []# 将计算的替代路径清空

    '''
    # Iterate over each node in the original route
    alternate_route.append(base_route[0])
    for i in range(len(base_route) - 1): # 沿原路径上的节点去寻找替代的路由
        # u = original_route[i]
        # v = original_route[i + 1]

        u = base_route[i]
        v = base_route[i + 1]
        # Check if the edge between u and v is present in the alternate graph
        if alternate_graph.has_edge(u, v) and alternate_graph.edges[u,v]['weight'] != float('inf'):
            # If the edge is present, add it to the alternate route
            alternate_route.append(v)
        else:
            # If the edge is not present, find an alternate path between u and v in the alternate graph
            try:
                # Find the shortest path between u and v in the alternate graph
                shortest_path = nx.shortest_path(alternate_graph, u, v, 'weight')
                if len(shortest_path) > 5: # 如果最短路径的跳数超过5跳则重新选择最短路计算的源节点
                    # u = base_route[i]
                    v = base_route[i+2]
                # Append the shortest path (excluding the first node, which is u) to the alternate route
                else:
                    alternate_route += shortest_path[1:]

            except nx.NetworkXNoPath:
                # If no path is found between u and v, return None to indicate that no alternate route exists
                return None
    '''
    dup = find_duplicates(alternate_route)
    if dup:
        print('当前计算的最短路径存在重复节点\n')
    return alternate_route

def find_local_route(G, original_route, faulty_node):
    alternate_route = []
    max_hop = 5
    # Create a copy of the original graph
    alternate_graph = G.copy()
    # Remove the faulty node from the alternate graph
    alternate_graph.remove_node(faulty_node)
    # 确定故障节点在原路径中的位置
    faulty_index = original_route.index(faulty_node)
    pred_nodes = original_route[ :faulty_index] # 业务原路由故障节点的前继节点
    pred_nodes_flip = pred_nodes[::-1] #将前继节点中的元素进行翻转
    succ_nodes = original_route[faulty_index+1: ] # 业务原路由故障节点的后继节点

    available_path = []
    for node in pred_nodes[]+succ_nodes[]:
        set_node_adj_weight(node, 'faulty')  # 将其余节点的邻接链路设置为inf

    for i in range(len(pred_nodes)):
        source = pred_nodes[i]
        set_node_adj_weight(source, 'operational')

        for j in range(len(succ_nodes)):
            target = succ_nodes[j]
            set_node_adj_weight(target, 'operational')

            # res_paths = list(islice(nx.shortest_simple_paths(alternate_graph, source, destination, 'weight'), 5))
            sub_path = nx.shortest_path(alternate_graph, source, target, 'weight')
            weight_sum = sum(alternate_graph[i][j]['weight'] for i, j in zip(sub_path, sub_path[1:]))
            if weight_sum != float('inf') and len(sub_path) < max_hop+i+j:
                available_path.append(sub_path)
                break

    return available_path


def find_duplicates(lst):
    # Create an empty set for tracking duplicates
    duplicates = set()
    # Create an empty set for tracking seen elements
    seen = set()
    for element in lst:
        if element in seen:
            duplicates.add(element)
        else:
            seen.add(element)
    return duplicates

def find_local_route_2(G_sample, app_original_path, app_demand, faulty_node, source, destination ):
    # 计算Local策略【尽量重用原路径】下的重路由
    # 在计算子路径之前，将现有的可用节点的相邻链路的权重设置为无穷
    K = 5
    max_hops = 6
    original_path_length = len(app_original_path)
    faulty_index = app_original_path.index(faulty_node)
    route_source = source
    route_destination = destination
    available_subpaths = [] # 计算符合条件的候选路径集合

    for n in app_original_path:
        if n != source and n != destination:  # G_sample.nodes[n]['alive'] == 1
            set_node_adj_weight(G_sample, n, 'faulty')

    if faulty_index == 0:  # 如果是源节点故障
        for i in range( original_path_length - 2): # 利用原路径多的那一侧
            optional_subpaths = list(islice(nx.shortest_simple_paths(G_sample, route_source, route_destination, 'weight'), K))
            # available_path = [] # 存储链路权重不为inf的最短路径
            for path in optional_subpaths:
                if faulty_node in path: # 如果新路径中存在故障节点
                    continue
                else:
                    # weight_sum = sum(G_sample[i][j]['weight'] for i, j in zip(path, path[1:]))
                    # if weight_sum != float('inf') and len(path) < max_hops+i:
                    if len(path) < max_hops + i:
                        available_subpaths.append(path)
            if available_subpaths:
                break
            else:
                route_destination = app_original_path[i+2]
                set_node_adj_weight(G_sample, route_destination, 'operational')

    elif faulty_index == original_path_length - 1:
        for i in range( original_path_length - 2): # 利用原路径多的那一侧
            optional_subpaths = list(islice(nx.shortest_simple_paths(G_sample, route_source, route_destination, 'weight'), K))
            # available_path = [] # 存储链路权重不为inf的最短路径
            for path in optional_subpaths:
                if faulty_node in path: # 如果新路径中存在故障节点
                    continue
                else:
                    # weight_sum = sum(G_sample[i][j]['weight'] for i, j in zip(path, path[1:]))
                    # if weight_sum != float('inf') and len(path) < max_hops + i:
                    if len(path) < max_hops + i:
                        available_subpaths.append(path)
            if available_subpaths:
                break
            else:
                route_source = app_original_path[faulty_index-i-2]
                set_node_adj_weight(G_sample, route_source, 'operational')
    else:
        for i in range( max(faulty_index-2, original_path_length - faulty_index-2)): # 利用原路径多的那一侧
            # print('当前的循环次数i={}'.format(i))
            optional_subpaths = list(islice(nx.shortest_simple_paths(G_sample, route_source, route_destination, 'weight'), K))
            # available_path = [] # 存储链路权重不为inf的最短路径
            for path in optional_subpaths:
                if faulty_node in path: # 如果新路径中存在故障节点
                    # print('故障节点{}在计算得到的路径中{}'.format(faulty_node,path))
                    continue
                else:
                    # weight_sum = sum(G_sample[i][j]['weight'] for i, j in zip(path, path[1:]))
                    # if weight_sum != float('inf') and len(path) < max_hops+i:
                    if len(path) < max_hops + i:
                        available_subpaths.append(path)
                        # print('可用的子路径为{}'.format(path))
            if available_subpaths:
                # print('子路径计算成功，为{}'.format(available_subpaths))
                break # 跳出循环
            elif faulty_index >= original_path_length/2: # 如果当前故障节点位于原业务路由的右侧，则将搜索的destination往左移，尽量重用右侧的路径
                route_source = app_original_path[faulty_index-i-2]
                set_node_adj_weight(G_sample, route_source, 'operational')
            else:
                route_destination = app_original_path[faulty_index+i+2]
                set_node_adj_weight(G_sample, route_destination, 'operational')

    new_subpath = find_available_path(G_sample, app_demand, available_subpaths, faulty_node) # 这只能找到子路径上的路径带宽是否满足

    return new_subpath
def set_node_adj_weight(G, node, strategy):
    if strategy == 'operational':
        adj_nodes = list(G.adj[node])
        for adj in adj_nodes:
            G.adj[node][adj]['weight'] = 1  # 将节点邻接的边的权重设置为1

    if strategy == 'faulty':
        adj_nodes = list(G.adj[node])
        for adj in adj_nodes:
            G.adj[node][adj]['weight'] = float('inf')  # 将节点邻接的边的权重设置为无穷大


def calculate_local_route(G, original_route, faulty_node, access_list, exit_list):

    available_path = []
    max_hop = 5
    # Create a copy of the original graph
    alternate_graph = G.copy()
    # Remove the faulty node from the alternate graph
    alternate_graph.remove_node(faulty_node)
    access = access_list.copy()
    exit = exit_list.copy()
    # 确定故障节点在原路径中的位置
    faulty_index = original_route.index(faulty_node)
    pred_nodes, succ_nodes = [], []

    if faulty_index == 0:
        access.remove(faulty_node)
        if len(access) > 0:  # 如果业务有其余可以接入的节点list
            available_access_list = []  # 当前演化态下存储可接入节点的list
            for n in access:
                if n not in original_route:  # 避免从已有的路径中选择source节点
                    available_access_list.append(n)
            if available_access_list:  # 如果存在可接入的节点list
                new_source = random.choice(available_access_list)
                pred_nodes = [new_source]
                succ_nodes = original_route[faulty_index+1:]

    elif faulty_index == len(original_route)-1:
        exit.remove(faulty_node)
        if len(exit) > 0:
            available_exit_list = []
            for n in exit:
                if n not in original_route:
                    available_exit_list.append(n)
            if available_exit_list:
                new_destination = random.choice(available_exit_list)
                succ_nodes = [new_destination]
                pred_nodes = original_route[:faulty_index]
    else:
        pred_nodes = original_route[:faulty_index]  # 业务原路由故障节点的前继节点
        succ_nodes = original_route[faulty_index + 1:]  # 业务原路由故障节点的后继节点

    pred_nodes_flip = pred_nodes[::-1]  # 将前继节点中的元素进行翻转

    for node in pred_nodes+succ_nodes:
        set_node_adj_weight(alternate_graph, node, 'faulty')  # 将其余节点的邻接链路设置为inf
    flag = 0
    for i in range(len(pred_nodes_flip)):
        source = pred_nodes_flip[i]
        set_node_adj_weight(alternate_graph, source, 'operational')

        for j in range(len(succ_nodes)):
            target = succ_nodes[j]
            set_node_adj_weight(alternate_graph, target, 'operational')

            # res_paths = list(islice(nx.shortest_simple_paths(alternate_graph, source, destination, 'weight'), 5))
            sub_path = nx.shortest_path(alternate_graph, source, target, 'weight')
            weight_sum = sum(alternate_graph[i][j]['weight'] for i, j in zip(sub_path, sub_path[1:]))
            if weight_sum != float('inf') and len(sub_path) < max_hop + i + j:
                available_path.append(sub_path)
                dup = find_duplicates(sub_path)
                if dup:
                    print('当前的链路节点计算重复')
                flag += 1

        if flag > 5: # 设定重路由发起的次数
            break


    return available_path

if __name__ == '__main__':
    # m = 4
    # res = calculate_SINR_outage_probability(m)
    m = 4
    gamma = 1
    noise_power = 0.0001
    d_ij = 30
    d_interferer_list = [ 35, 20]
    alpha = 2.5
    Pt = 1.5
    # s = symbols('s', real=True)
    k = (m*gamma*noise_power)/(Pt* pow(d_ij, -alpha))
    b = gamma*pow((d_ij/25),-alpha)
    s = symbols('s')
    f = function_SINR(s, d_ij, d_interferer_list, noise_power)
    # r = diff(f, s, 4).subs({s:1})
    # print(r)

    res = calculate_SINR_outage_probability(m, d_ij, d_interferer_list, noise_power)
    print('链路的故障概率为{}'.format(res))

    # res = diff(f,s,2)

    # line_List = [['aa', 'bb', 'cc'], ['dd', 'ee', 'ff'], ['gg', 'hh']]
    # num_list = [[69, 31, 98], [61, 97, 78] ]
    #
    # caseslist = RecursionFunc(line_List[0], line_List[1:])
    # numslist = RecursionFunc(num_list[0], num_list[1:])
    # for num in numslist:
    #     print(num)
    #     print('*******\n')
    # G = nx.random_graphs.erdos_renyi_graph(100, 0.2)
    # N = 10
    # Nodes = list(G)
    # start_time = time.time()
    # for i in range(N):
    #     source = random.choice(Nodes)
    #     destination = random.choice(Nodes)
    #     print('源节点为{}，宿节点为{}'.format(source, destination))
    #     path = find_path(G, source, destination)
    #     print('业务路径为{}'.format(path))
    #
    # end_time = time.time()
    # print('for循环{}次的总耗时为{}s'.format(N, end_time-start_time))
    #
    # t1 = time.time()
    # multi_threading_func(N, G)
    # t2 = time.time()
    # print('多线程并发{}次的总耗时为{}s'.format(N, t2-t1))



''' 临时保存演化规则中路由部分的代码
    if strategy == 'Separate':
        if app_failtime == 0: # 如果业务上一时刻未发生故障
            # 如果是链路故障触发的重路由，则把业务原有路径上的可用路径挑选出来
            app_avail_path = [x for x in app_original_path if x not in link_fail_list]  # 统计业务原有路径上未故障的链路
        else: # 否则业务的可用路径为空集，避免业务在恢复时触发重路由将其原路径的权重设置为不可用而导致无法恢复上线
            app_avail_path = [] # 因为业务在故障发生时已经将原路径上占用的带宽释放

        # 计算业务od间的K条最短路
        for link in app_avail_path: # 将原业务路径集合中的链路权重设置为‘inf'，从而避免选路时计算这些路径；
            G_sample.adj[link[0]][link[1]]['weight'] = float('inf')
        # 设置路由发起的源宿节点对

        path_tmp = k_shortest_paths( K_max, G_sample, source, destination, 'weight') # Generate k-shortest paths in the graph G from source to target.

        i = 0
        while not path_tmp: # 如果计算出来的重路由路径集为空集的话
            link = app_avail_path[i]
            G_sample.adj[link[0]][link[1]]['weight'] = 1 # 重用业务原路径集中的第i条链路
            path_tmp = k_shortest_paths(K_max, G_sample, source, destination, 'weight')
            i += 1
            if i >= len(app_avail_path):
                break
        candidate_paths = path_tmp # 将计算出来的路径赋值给业务路径

    if strategy == 'Repetition':
        # G_sample = copy.deepcopy(G)

        # P_space = G_sample.neighbors(link_fail_list[0][0])

        # 计算业务od间的K条最短路
        # for link in app_avail_path: # 将原业务路径集合中的链路权重置为0，尽量重用
        # G_sample.adj[link[0]][link[1]]['weight'] = 1

        # PSR,QSR 故障链路的前/后搜索点
        PSR_index, QSR_index = [], [] # 存储故障链路对应在app_path中的下标位置

        # 前/后搜索点在业务路径app_path中的下标
        if app_failtime == 0: # 如果业务最近第一次故障
            for link in link_fail_list:
                if app_path.index(link[0]) < app_path.index(link[1]):
                    PSR_index.append(app_path.index(link[0]))
                    QSR_index.append(app_path.index(link[1]))
                else:
                    PSR_index.append(app_path.index(link[1]))
                    QSR_index.append(app_path.index(link[0]))
        else:
            PSR_index = [0]
            QSR_index = [-1]
        # 最前和最后的搜索点下标
        Min_PSR_index = min(PSR_index)
        Max_QSR_index = max(QSR_index)
        path_tmp = k_shortest_paths(K_max, G_sample, app_path[Min_PSR_index], app_path[Max_QSR_index], 'weight')  # Generate k-shortest paths in the graph G from source to target.
        while (not path_tmp and Min_PSR_index >= 0 and  Max_QSR_index <= len(app_path) - 1):
            Min_PSR_index = Min_PSR_index - 1
            Max_QSR_index = Max_QSR_index + 1
            path_tmp = k_shortest_paths(K_max, G_sample, app_path[Min_PSR_index], app_path[Max_QSR_index], 'weight')

        # 业务重路由链路拼接
        for j in range(0, len(path_tmp)): # 对K最短路径集中的链路进行拼接
            Min_Reroute_index = Min_PSR_index # 重路由发起的左端点
            Max_Reroute_index = Max_QSR_index # 重路由发起的右端点
            for x in path_tmp[j]: # 依次遍历候选路径中的节点
                if x in app_path:  # 重路由路径中含原路径最靠前/后节点的下标
                    Min_Reroute_index = min(app_path.index(x), Min_Reroute_index)
                    Max_Reroute_index = max(app_path.index(x), Max_Reroute_index)
            # Min_Reroute_index = min(Min_Reroute_index, Min_PSR_index)
            # print(Min_Reroute_index, Min_PSR_index)
            # Max_Reroute_index = max(Max_Reroute_index, Max_QSR_index)
            # 拼接的左右节点在新生成路径中的下标
            Min_Reroute_index_in_tmp = path_tmp[j].index(app_path[Min_Reroute_index])
            Max_Reroute_index_in_tmp = path_tmp[j].index(app_path[Max_Reroute_index])
            if Min_Reroute_index_in_tmp < Max_Reroute_index_in_tmp: # 如果在新生成路径中该左端点的索引值小于右端点的索引值
                slice = path_tmp[j][Min_Reroute_index_in_tmp:Max_Reroute_index_in_tmp+1]
                combine_path = app_path[0:Min_Reroute_index] + slice + app_path[Max_Reroute_index+1:]
            else:
                slice = path_tmp[j][Max_Reroute_index_in_tmp:Min_Reroute_index_in_tmp+1] # 不能同时对list做切片和翻转操作
                slice.reverse() # 翻转即可，没有返回值；
                print('业务的原路径为{}'.format(app_path))
                print('尽量重用的路径为{}'.format(path_tmp[j]))
                print('切片的路径为{}'.format(slice))
                combine_path = app_path[0:Min_Reroute_index] + slice + app_path[Max_Reroute_index+1:]

            # print("-------------/n")
            # print(path_tmp[j][Min_Reroute_index_in_tmp:Max_Reroute_index_in_tmp+1])

            for i in range(len(combine_path)-1):
                if not G_sample.has_edge(combine_path[i], combine_path[i+1]):
                    print('该链路在网络中不存在，{}'.format((combine_path[i], combine_path[i+1])))
                    print('计算的组合链路为{}'.format(combine_path))
                    print('业务的原路径为{}'.format(app_path))
                    print('尽量重用的路径为{}'.format(path_tmp[j]))

            candidate_paths.append(combine_path)


    candidate_paths_hash = {a: len(candidate_paths[a]) for a in range(len(candidate_paths))} # 存储路径id和路径的长度，便于根据路径长度进行排序
    candidate_paths_sorted = sorted(candidate_paths_hash.items(), key=lambda x: x[1], reverse=False)

    # 确定最短路径上的节点是否满足业务的带宽需求
    for each in candidate_paths_sorted:
        path_id = each[0]
        path = candidate_paths[path_id]
        path_exist = 1 #先假设路径存在
        # 计算路径的存在概率
        for i in range(len(path)-1):
            delta = random.random()
            link_fail_rate = G_sample.adj[path[i]][path[i+1]]['fail_rate']
            # print('生成的随机数为{}'.format(delta))
            # print('链路的故障率为{}\n'.format(link_fail_rate))

            if delta < link_fail_rate: # 如果随机数大于该链路的中断率，则该链路不存在，需要重新选路
                reroute_times += 1
                path_exist = 0
                break
        if path_exist != 0:
            # 判断当前路径上节点的容量是否满足业务需求
            cap_available = min([G.nodes[n]['cap']-G.nodes[n]['load']] for n in path)  # 计算业务路径上的剩余可用容量
            if cap_available >= app_demand:
                new_app_path = path  # 如果当前链路满足业务的带宽需求，则将该路径选择为业务的新路径
            else:
                # 否则表示重新进行路由
                reroute_times += 1
'''