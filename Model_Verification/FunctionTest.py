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

def function_SINR(s, d_ij, d_interferer_list):
    m = 4
    gamma = 1
    W = 0.0001
    # d_ij = 20
    # d_interferer_list = [35]
    alpha = 2
    Pt = 1.5
    # s = symbols('s', real=True)
    k = (m * gamma * W) / (Pt*pow(d_ij, -alpha))

    y1 = 1
    for d_vj in d_interferer_list:
        b = gamma * pow((d_vj / d_ij), -alpha)
        print('b 的值为{}'.format(b))
        y1 *= pow( (1+s*b), -m)
    y2 = exp(-k*s)  * y1

    return y2


def calculate_SINR_outage_probability(m, d_ij, d_interferer_list):
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
        bb = diff(function_SINR(s, d_ij, d_interferer_list), s , n).subs({s:1})
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


if __name__ == '__main__':
    # m = 4
    # res = calculate_SINR_outage_probability(m)
    m = 4
    gamma = 1
    W = 0.0001
    d_ij = 30
    d_interferer_list = [15, 35]
    alpha = 2.5
    Pt = 1.5
    # s = symbols('s', real=True)
    k = (m*gamma*W)/(Pt* pow(d_ij, -alpha))
    b = gamma*pow((d_ij/25),-alpha)
    s = symbols('s')
    f = function_SINR(s, d_ij, d_interferer_list)
    # r = diff(f, s, 4).subs({s:1})
    # print(r)

    res = calculate_SINR_outage_probability(m, d_ij, d_interferer_list)
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