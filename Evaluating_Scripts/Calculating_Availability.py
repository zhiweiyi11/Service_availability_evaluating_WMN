#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   ：Service_availability -> Calculating_Availability
@IDE    ：PyCharm
@Author ：Yi Zhiwei
@Date   ：2023/3/13 11:32
@Desc   ： 计算服务的可用度
=================================================='''
import numpy as np
import networkx as nx
import copy
import time
import os

from multiprocessing import  cpu_count

import pandas as pd

from Evolution_Model.Evolution_Objects import *
from Evolution_Model.Evolution_Conditions import *
from Evolution_Model.Evolution_Rules import *
from Evolution_Model.Application_request_generating import *
from concurrent.futures import ThreadPoolExecutor, wait
from concurrent.futures import ProcessPoolExecutor

from multiprocessing.pool import ThreadPool as Pool


def single_availability(App_set, Switch_time, Survival_time, T):
    # 计算单次网络演化时的业务可用度(仅考虑中断时长)
    Apps_avail = {}  # 业务单次演化的可用度
    Apps_loss = {}

    for app_id, app_val in App_set.items():

        app_out = app_val.outage
        rp_t = 0
        sw_t = 0
        for i in app_out['reroute']:
            # 如果重路由的时长超过业务的生存时间，则计入业务的中断时长
            if i*Switch_time > Survival_time:
                sw_t += i*Switch_time
        for j in app_out['repair']:
            rp_t += j

        app_unavail = sw_t + rp_t * 3600
        app_loss = (app_unavail * app_val.demand) / 3600 # 每小时的期望服务负载损失
        app_avail = 1 - app_unavail/(3600*T) # 在T时间间隔内的平均服务可用度
        Apps_avail[app_id] = app_avail #
        Apps_loss[app_id] = app_loss
        # 测试业务可用度
        # if app_avail < 0.99 :
        #     print('业务的id为{}，业务的故障时间为{}'.format(Apps[app_id].id, Apps[app_id].outage ))

    return Apps_avail, Apps_loss

def calculateAvailability(T, G, App_dict, MTTF, MLife, MTTR, switch_time, switch_rate, survival_time):
    # 生成网络演化对象和演化条件，调用演化规则，来模拟网络的演化过程(Sub_service为服务部署的节点集合)
    # 1: 生成网络的演化对象
    start_time = time.time()
    routing_th = 10 # 重路由的次数
    G_tmp, App_tmp = copy.deepcopy(G), copy.deepcopy(App_dict)
    Nodes = list(G)

    # 2: 生成网络的演化条件
    evo_conditions = cond_func(G_tmp, MTTF, MLife, MTTR, T)  # 各链路的MTTF、MTTR具体值在函数中根据对应的分布进行修改
    time_list = evo_conditions.index.tolist()  # 网络发生演化的时刻
    print('生成的演化态数量为{}'.format(len(time_list)))

    # 3: 触发网络的演化规则
    App_interupt = [] # 发生中断的业务集合
    for i in range(len(time_list)):
        # print('当前为第{}次演化'.format(i))
        evo_time = time_list[i]
        App_fail = []  # 当前演化时刻下故障的业务集合
        nodes_fail = evo_conditions.iloc[i]['fail']
        nodes_reco = evo_conditions.iloc[i]['repair']
    # 3.1: 首先根据修复的构件,对等待上线的业务进行恢复
        # 3.1.1: 针对网络的action
        for n in nodes_reco:
            G_tmp.nodes[n]['alive'] = 1  # 置节点的状态为1，表示存活
            # 将节点相邻节点的链路权重设置为1，表示节点恢复上线
            adj_nodes = list(G.adj[n])
            for adj in adj_nodes:
                G_tmp.adj[n][adj]['weight'] = 1
        # 3.1.2: 针对业务的action
        if App_interupt:
            # action1: 对业务进行按优先级排序
            App_hash = {App_tmp[a].id: App_tmp[a].SLA for a in App_interupt}
            App_sorted = sorted(App_hash)  # 返回列表，按字典的value值从小到大排序：即业务优先级值越小越优先

            for app_id in App_sorted:
            # action2: 对业务进行路径计算
                app = App_tmp[app_id]
                app_path = app.path
                new_app_path, reroute_times = path_reroute(G_tmp, app.access, app.exit, app_path, app.demand, app.str, [], routing_th)
            # action3: 对业务进行重新部署
                if new_app_path:  # 如果业务的新路径不为空
                    app.path = new_app_path
                    app.app_deploy_node(G_tmp)
                    app.outage['repair'].append(evo_time - app.fail_time) # 这里仅考虑业务中断的时长,不考虑重路由的次数
                    app.fail_time = 0  # 将业务的故障时刻清零
                    App_interupt.remove(app_id)  # 从待恢复的业务集合中移除完成上线的业务id

    # 3.2: 然后根据故障的构件,对故障的业务进行恢复

        # 3.2.1: 针对网络的action
        for n in nodes_fail:
            G_tmp.nodes[n]['alive'] = 0  # 置节点的状态为0，表示失效
            # 将节点相邻节点的链路权重设置为无穷大，表示节点故障下线
            adj_nodes = list(G.adj[n])
            for adj in adj_nodes:
                G_tmp.adj[n][adj]['weight'] = float('inf')
            failed_apps = G_tmp.nodes[n]['app_dp']
            App_fail += failed_apps
        # 3.2.2 针对业务的action
        if App_fail:
            # action1: 对业务进行按优先级排序
            App_fail = list(set(App_fail))  # 去掉重复统计的业务id（即可能一条业务的多个链路/节点中断）
            App_hash = {App_tmp[a].id: App_tmp[a].SLA for a in App_fail}
            App_sorted = sorted(App_hash)  # 返回列表，按字典的value值从小到大排序：即业务优先级值越小越优先 # sorted(App_hash, key=lambda x: x[1], reverse=False)

            for app_id in App_sorted:
            # action2: 对业务进行路径计算
                app = App_tmp[app_id]
                app.app_undeploy_node(G_tmp)  # 释放业务之前的路径映射
                app_path = app.path
                # 区别在重路由计算的输入参数不一样
                new_app_path, reroute_times = path_reroute(G_tmp, app.access, app.exit, app_path, app.demand, app.str, nodes_fail, routing_th)
            # action3: 对业务进行重新部署
                rand_seed = random.random() # 生成随机数来作为路径倒换的判断
                if new_app_path and rand_seed <= switch_rate:  # 如果业务的新路径不为空
                    app.path = new_app_path
                    app.app_deploy_node(G_tmp)
                    app.outage['reroute'].append(reroute_times)  # 这里加入每次业务重路由的次数,最后联合业务的生存时间做判断
                    app.fail_time = 0  # 将业务的故障时刻清零
                else:
                    App_interupt.append(app_id)  # 从待恢复的业务集合中移除完成上线的业务id
                    app.fail_time = evo_time

    print('当前演化下业务可用度计算完成\n')
    onetime_availability = single_availability(App_tmp, switch_time, survival_time, T)

    return onetime_availability

def Apps_Availability_Count(N, func_name, T, G, App, MTTF, MLife, MTTR, switch_time, switch_rate, survival_time):
    # 开启多线程进行并行计算
    # 调用 ThreadPoolExecutor 类的构造器创建一个包含20条线程的线程池
    multi_avail = pd.DataFrame(index=list(App.keys())) # 存储N次演化下各次的业务可用度结果
    multi_loss = pd.DataFrame(index=list(App.keys())) # 存储N次演化下各次业务的带宽损失结果

    executor = ThreadPoolExecutor(max_workers=10)
    result = []
    RES = []
    def get_result(future):
        res = future.result()  # 获取该 Future 代表的线程任务最后返回的结果,若Future代表的线程任务还未完成，该方法会阻塞当前线程，timeout参数指定最多阻塞多少秒
        return res[0]

    for i in range(N):
        # 调用 ThreadPoolExecutor 对象的 submit() 方法来提交线程任务
        task = executor.submit(func_name, T, G, App, MTTF, MLife, MTTR, switch_time, switch_rate, survival_time) # 将 fn 函数提交给线程池：*args 代表传给 fn 函数的参数，*kwargs 代表以关键字参数的形式为 fn 函数传入参数
        print('当前第{}次调用线程'.format(i))
        result.append(task)
        # result.append(task.add_done_callback(get_result)) # 当线程任务完成后，程序会自动触发该回调函数，并将对应的 Future 对象作为参数传给该回调函数


    wait(result) # 阻塞主进程，直到所有的子线程都执行完
    # 关闭线程池
    # executor.shutdown(wait=True)

    return RES



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

def Apps_Availability_MC(N,T, App_set, G, MTTF, MLife, MTTR, switch_time, switch_rate, survival_time):
    # 计算业务可用度的主函数，采用蒙特卡洛方法
    multi_avail = pd.DataFrame(index=list(App_set.keys())) # 存储N次演化下各次的业务可用度结果
    multi_loss = pd.DataFrame(index=list(App_set.keys())) # 存储N次演化下各次业务的带宽损失结果

    for n in range(N):
        result = calculateAvailability(T, G, App_set, MTTF, MLife, MTTR, switch_time, switch_rate, survival_time)
        multi_avail.loc[:, n + 1] = pd.Series(result[0])  # 将单次演化下各业务的可用度结果存储为dataframe中的某一列(index为app_id)，其中n+1表示列的索引
        multi_loss.loc[:, n + 1] = pd.Series(result[1])
        # print('\n 当前为第{}次蒙卡仿真'.format(n))

    return multi_avail, multi_loss

def save_results(origin_df, file_name):
    # 保存仿真的数据
    # 将dataframe中的数据保存至excel中
    # localtime = time.asctime(time.localtime(time.time()))
    time2 = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M') # 记录数据存储的时间
    sys_path = os.path.abspath('..')  # 表示当前所处文件夹上一级文件夹的绝对路径

    with pd.ExcelWriter(sys_path + r'.\Results_saved\{}_time{}.xlsx'.format(file_name, time2)) as xlsx: # 将紧跟with后面的语句求值给as后面的xlsx变量，当with后面的代码块全部被执行完之后，将调用前面返回对象的exit()方法。
        origin_df.to_excel(xlsx, sheet_name='app_avail', index=False) # 不显示行索引
        print('数据成功保存')


if __name__ == '__main__':
    # 网络演化对象的输入参数；
    ## 网络层对象
    Topology = 'Random'
    Node_num, App_num = 100, 50
    Capacity = 50
    Demand = np.random.normal(loc=10, scale=2, size=App_num)  # 生成平均值为5，标准差为1的带宽的正态分布
    Area_width , Area_length = 250, 150
    Area_size = (250,150)

    TX_range = 50  # 传输范围为区域面积的1/5时能够保证网络全联通
    CV_range = 30
    Coordinates = generate_positions(Node_num, Area_width, Area_length)
    # Demand = list(map(int, Demand)) # 将业务的带宽需求换成整数
    ## 业务层对象
    grid_size = 5
    traffic_th = 0.5
    Priority = np.linspace(start=1, stop=5, num=5, dtype=int)
    ratio_str = 0.5  # 尽量分离和尽量重用的业务占比
    Strategy_P = ['Global'] * int(App_num * (1 - ratio_str))
    Strategy_S = ['Local'] * int(App_num * ratio_str)
    Strategy = Strategy_S + Strategy_P

    # 演化条件的参数
    T = 8760
    MTTF, MLife = 1000, 800
    MTTR = 2

    ## 重路由相关的参数
    switch_time = 10
    switch_rate = 0.99
    survival_time = 3*switch_time # 允许的最大重路由次数为5次

    # 初始化网络演化对象
    start_time = time.time()
    G, App = init_func(Area_size, Node_num, Topology, TX_range, CV_range, Coordinates, Capacity, grid_size,  App_num, traffic_th, Demand, Priority, Strategy)
    # 生成网络演化条件
    AppAvailability_results = calculateAvailability(T, G, App, MTTF, MLife, MTTR, switch_time, switch_rate, survival_time)
    end_time = time.time()
    print('\n 单次网络演化的时长为{}s \n'.format(end_time-start_time))

    # 测试多进程/线程运行结果是否正确
    N = 10
    pool_num = 6
    args = [T, G, App, MTTF, MLife, MTTR, switch_time, switch_rate, survival_time]
    # Availability_Results = Apps_availability_func(N, args, pool_num)
    st1 = time.time()
    # RES = Apps_Availability_Count(N, calculateAvailability, T, G, App, MTTF, MLife, MTTR, switch_time, switch_rate, survival_time)
    Res = Apps_availability_func(N, args, pool_num)
    et1 = time.time()
    print('\n 采用多进程计算{}次网络演化的时长为{}s \n'.format(N, et1 - st1))

    # 测试普通蒙卡的仿真效率
    st2 = time.time()
    Res2 = Apps_Availability_MC(N,T, App, G, MTTF, MLife, MTTR, switch_time, switch_rate, survival_time)
    et2 = time.time()
    print('\n 采用普通蒙卡计算{}次网络演化的时长为{}s \n'.format(N, et2 - st2))




