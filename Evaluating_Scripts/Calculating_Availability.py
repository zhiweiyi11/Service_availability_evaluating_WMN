#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   ：Service_availability -> Calculating_Availability
@IDE    ：PyCharm
@Author ：Yi Zhiwei
@Date   ：2023/3/13 11:32
@Desc   ： 计算服务的可用度
=================================================='''
import random
import threading

import numpy as np
import networkx as nx
import copy
import time
import os

from multiprocessing import  cpu_count

import pandas as pd
from numba.cuda.cudadrv.runtime import Runtime

from Evolution_Model.Evolution_Objects import *
from Evolution_Model.Evolution_Conditions import *
from Evolution_Model.Evolution_Rules import *
from Evolution_Model.Application_request_generating import *
from concurrent.futures import ThreadPoolExecutor, wait
from concurrent.futures import ProcessPoolExecutor
from Evaluating_Scripts.MTBF_sensitivity_analysis import *
from multiprocessing.pool import ThreadPool as Pool


def single_availability(App_set, T, beta, demand_threshold):
    # 计算单次网络演化时的业务可用度(仅考虑中断时长)
    Apps_time_avail = {}  # 仅考虑中断时长的服务可用度
    Apps_performance_avail = {} # 考虑业务实际负载的可用度
    Single_app_avail = {}
    Whole_network_avail = 0
    Totoal_traffic_flow = 0

    for app_id, app_val in App_set.items():
        app_demand = app_val.demand
        Totoal_traffic_flow += app_demand
        app_out = app_val.outage
        rp_t = 0
        sw_t = 0
        dw_t = 0
        perf_loss = 0
        for i in app_out['reroute']:
            # 如果重路由的时长超过业务的生存时间，则计入业务的中断时长
            sw_t += i
        for j in app_out['repair']:
            rp_t += j

        for k in app_out['degradation']:
            for key, value in k.items(): #这里的key为业务降级时的负载值，value为业务降级的时长
                if key < demand_threshold:
                    dw_t += value
                else:
                    perf_loss += (app_demand - key) * value *3600 # 转换为以秒s为单位


        app_unavail = sw_t + rp_t *3600 + dw_t *3600
        app_time_avail = 1 - app_unavail/(3600*T) # 在T时间间隔内的平均服务可用度
        # 服务的性能可用度=1-(总性能损失值)/总期望的性能
        app_performance_avail = 1 - ((app_unavail * app_demand) + (perf_loss ))/ (3600*T*app_demand)
        # if app_performance_avail < 0.9:
        #     print('业务{}的性能可用度为{}'.format(app_id, app_out['degradation']))
        #     # print('业务{}的修复时间为{}'.format(app_id, app_out['repair']))

        Apps_time_avail[app_id] = app_time_avail #
        Apps_performance_avail[app_id] = app_performance_avail
        Single_app_avail[app_id] = beta*app_time_avail + (1-beta)*app_performance_avail
        # if Single_app_avail[app_id] < 0.9:
        #     print('业务{}的性能可用度为{}'.format(app_id, app_out['degradation']))
        #     print('业务{}的修复时长为{}'.format(app_id, app_out['repair']))


    # 计算整网的平均服务可用度
    for app_id, app_val in App_set.items():
        weight = app_val.demand / Totoal_traffic_flow # 权重为业务的带宽请求占整网总业务带宽请求的比例
        Whole_network_avail += weight * Single_app_avail[app_id]

    return Single_app_avail, Whole_network_avail

def calculateAvailability(T, G, App_dict, MTTF, MLife, MTTR,  detection_rate, message_processing_time, path_calculating_time, beta_list, demand_th):
    # 生成网络演化对象和演化条件，调用演化规则，来模拟网络的演化过程(Sub_service为服务部署的节点集合)
    # 1: 生成网络的演化对象
    routing_th = 10 # 重路由的次数阈值
    # G_tmp, App_tmp = copy.deepcopy(G), copy.deepcopy(App_dict)
    G_tmp , App_tmp  = G, App_dict
    Nodes = list(G)

    # 2: 生成网络的演化条件
    evo_conditions = cond_func(G_tmp, MTTF, MLife, MTTR, T)  # 各链路的MTTF、MTTR具体值在函数中根据对应的分布进行修改
    time_list = evo_conditions.index.tolist()  # 网络发生演化的时刻
    print('生成的演化态数量为{}'.format(len(time_list)))

    # 3: 触发网络的演化规则
    App_interrupt = [] # 发生中断的业务集合
    for i in range(len(time_list)):
        # print('当前为第{}次演化'.format(i))
        evo_time = time_list[i]
        # print('\n 当前的演化时刻为{}'.format(evo_time))
        nodes_fail = evo_conditions.iloc[i]['fail']
        # print('故障节点为{}'.format(nodes_fail))
        nodes_reco = evo_conditions.iloc[i]['repair']
        # 3.1: 首先根据修复的构件,对等待上线的业务进行恢复
        component_repair(G_tmp, App_tmp, App_interrupt, evo_time, nodes_reco)
        # if App_interrupt:
        #     print('仍然中断的app为{}'.format(App_interrupt))
        #     for app_id in App_interrupt:
        #         print('仍然中断的app {} 的故障时刻为{}'.format(app_id, App_tmp[app_id].fail_time))

        # 3.2: 然后根据故障的构件,对故障的业务进行恢复
        apps_fault = component_failure(G_tmp, App_tmp, evo_time, nodes_fail) # 读取发生故障的业务
        # print('故障的业务集合为{} '.format(apps_fault))

        # action1: 对故障业务进行故障检测的action
        App_reroute, App_repair = app_fault_detect(detection_rate, apps_fault)
        App_interrupt += App_repair # 将未检测到故障的业务加入中断的业务集合中
        # action2: 对业务进行按优先级排序
        App_hash = {App_tmp[a].id: App_tmp[a].SLA for a in App_reroute}
        App_reroute_sorted = sorted(App_hash)  # 返回列表，按字典的value值从小到大排序：即业务优先级值越小越优先 # sorted(App_hash, key=lambda x: x[1], reverse=False)
        # print('需要重路由的app 为{}'.format(App_reroute_sorted))
        G_tmp.generate_link_state()  # 根据链路的故障率来确定网络链路的权重;(不能这样设置,这样会导致故障率高的链路就经常断,业务都路由到未故障的链路上去了)

        # action3: 对业务进行重路由和带宽分配
        App_successful_reroute = []
        for app_id in App_reroute_sorted: # 优先级高的业务先进行重路由和带宽分配
            app = App_tmp[app_id]
            app_original_path = app.path
            # print('\n app original path is {}'.format(app_original_path))
            recovery_parameters = [message_processing_time, path_calculating_time, len(App_reroute)]
            app_new_path, reroute_duration = path_reroute(G_tmp,  app.demand, app.access, app.exit, app_original_path, app.str, nodes_fail, recovery_parameters)
            # print('app _new path is {}'.format(app_new_path))
            if app_new_path: # 如果业务重路由成功
                # print('reroute successful app id is {}'.format(app_id))
                App_successful_reroute.append(app_id) # 记录重路由成功的业务id
                app.app_undeploy_node(G_tmp) # 解除业务到节点的映射
                app.app_undeploy_edge(G_tmp) # 在路径更新前需要同时解除业务到链路上的带宽映射
                app.fail_time = 0
                app.path = app_new_path
                # print('业务id {}的新路径为{}'.format(app_id, app.path))
                app.outage['reroute'].append(reroute_duration)
                app.app_deploy_node(G_tmp) # 将业务新的路径部署到节点上去

                # 以下是新改写的代码
                app_original_load = app.load
                app_new_load, degradation_duration, degradation_time = load_allocate(G_tmp, evo_time, app.path, app.demand, app_original_load, app.down_time)
                app.load = app_new_load
                if app_new_load == app.demand:
                    app.down_time = 0
                else:
                    app.down_time = evo_time
                if degradation_duration:  # 如果存在降级，则将降级时长及负载记录
                    app.outage['degradation'].append(degradation_duration)
                app.app_deploy_edge(G_tmp)


            else:
                App_interrupt.append(app_id)
                App_tmp[app_id].fail_time = evo_time # 上一个双保险,确保重路由不成功的app的故障时间被记录
                # print('重路由不成功的app为{} '.format(app_id))

        G_tmp.restore_link_state() # 将链路的权重还原为其非中断率【这里还需要修改】

    OneTime_SLA_availability, OneTime_whole_availability = [], []
    for beta in beta_list:
        onetime_availability = single_availability(App_tmp, T, beta, demand_th) # 单次演化的业务可用度结果
        OneTime_SLA_availability.append(onetime_availability[0])
        OneTime_whole_availability.append(onetime_availability[1])
    # print('当前演化下业务可用度计算完成 \n')
    # time.sleep(2)
    # print(threading.current_thread().name + '执行操作的业务可用度结果是={} \n'.format(onetime_availability[0][1] ))
    return OneTime_SLA_availability, OneTime_whole_availability


def Apps_Availability_Count(N, func_name, T, G, App, MTTF, MLife, MTTR, switch_time, switch_rate, survival_time):
    # 开启多线程进行并行计算
    # 调用 ThreadPoolExecutor 类的构造器创建一个包含20条线程的线程池
    multi_avail = pd.DataFrame(index=list(App.keys())) # 存储N次演化下各次的业务可用度结果
    multi_loss = pd.DataFrame(index=list(App.keys())) # 存储N次演化下各次业务的带宽损失结果
    res_avail, res_loss = [], []
    task = []
    # executor = ThreadPoolExecutor(max_workers=10)
    G_tmp, App_tmp = G, App
    with ThreadPoolExecutor(max_workers=200) as pool:
        for i in range(N):
            G_tmp, App_tmp = copy.deepcopy(G), copy.deepcopy(App) # 这里采用了深拷贝，会影响多线程的执行速率
            future = pool.submit(func_name, T, G_tmp, App_tmp, MTTF, MLife, MTTR, switch_time, switch_rate, survival_time)
            def get_result(future):
                # 定义获取结果的函数, 这样可以非阻塞地获取线程执行的结果
                res = future.result()  # 获取该 Future 代表的线程任务最后返回的结果,若Future代表的线程任务还未完成，该方法会阻塞当前线程，timeout参数指定最多阻塞多少秒
                res_avail.append(res[0]) # 因为结果是乱序的，所以只能先用list添加进来
                res_loss.append(res[1])

            future.add_done_callback(get_result) # 通过add_done_callback函数向线程池中注册了一个获取线程执行结果的函数get_result

        # print(future.done())
        # wait(task)  # 阻塞主进程，直到所有的子线程都执行完
    print('-------------主线程执行结束------------')
    # 关闭线程池
    # executor.shutdown(wait=True)
    return res_avail, res_loss

def Apps_Availability_MC(N,T, G, Apps,  MTTF, MLife, MTTR, detection_rate, message_processing_time, path_calculating_time, beta_list, demand_th):
    # 计算业务可用度的主函数，采用蒙特卡洛方法, 仅适用于单个beta的计算

    multi_single_avail = pd.DataFrame(index=list(Apps.keys())) # 存储N次演化下各次的业务可用度结果
    multi_whole_avail = pd.DataFrame(index= ['evo_times']) # 存储N次演化下整网业务的可用度结果

    for n in range(N):
        st_time = time.time()
        G_tmp = copy.deepcopy(G)
        App_tmp = copy.deepcopy(Apps)
        SLA_avail, whole_avail = calculateAvailability(T, G_tmp, App_tmp, MTTF, MLife, MTTR, detection_rate, message_processing_time, path_calculating_time, beta_list, demand_th)
        # print('当前第{}次循环业务的可用度为{}'.format(n, result[0][1]))
        multi_single_avail.loc[:, n + 1] = pd.Series(SLA_avail[0])  # 将单次演化下各业务的可用度结果存储为dataframe中的某一列(index为app_id)，其中n+1表示列的索引
        multi_whole_avail.loc[:, n + 1] = whole_avail[0]
        ed_time = time.time()
        print('\n 当前为第{}次蒙卡仿真, 仿真时长为{}s'.format(n, ed_time-st_time))

    return multi_single_avail, multi_whole_avail

def save_results(origin_df, file_name):
    # 保存仿真的数据
    # 将dataframe中的数据保存至excel中
    # localtime = time.asctime(time.localtime(time.time()))
    time2 = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M') # 记录数据存储的时间
    sys_path = os.path.abspath('..')  # 表示当前所处文件夹上一级文件夹的绝对路径

    with pd.ExcelWriter(sys_path + r'.\Results_saving\{}_time{}.xlsx'.format(file_name, time2)) as xlsx: # 将紧跟with后面的语句求值给as后面的xlsx变量，当with后面的代码块全部被执行完之后，将调用前面返回对象的exit()方法。
        origin_df.to_excel(xlsx, sheet_name='app_avail', index=False) # 不显示行索引
        print('数据成功保存')



if __name__ == '__main__':
    # 网络演化对象的输入参数；
    save_data = False # 是否保存节点坐标数据
    Node_num = 200
    Area_size = (250, 250)
    Area_width, Area_length = 250, 250
    Coordinates = generate_positions(Node_num, Area_width, Area_length, save_data)

    # TX_range = 50 # 传输范围为区域面积的1/5时能够保证网络全联通
    transmit_prob = 0.1 # 节点的传输概率
    transmit_power = 1.5  # 发射功率(毫瓦)，统一单位：W
    path_loss = 2  # 单位：无
    noise = pow(10, -11)  # 噪声的功率谱密度(毫瓦/赫兹)，统一单位：W/Hz, 参考自https://dsp.stackexchange.com/questions/13127/snr-calculation-with-noise-spectral-density
    bandwidth = 10 * pow(10, 6)  # 带宽(Mhz)，统一单位：Hz
    lambda_TH = 8 * pow(10, -1)  # 接收器的敏感性阈值,用于确定节点的传输范围
    # TX_range = pow((transmit_power / (bandwidth * noise * lambda_TH)), 1 / path_loss)
    TX_range = 30
    CV_range = 30  # 节点的覆盖范围

    # 业务请求的参数
    # App_num = 100
    # grid_size = 5
    # traffic_th = 1  # 业务网格的流量阈值
    # App_Demand = np.random.normal(loc=3, scale=1, size=App_num)  # 生成平均值为3，标准差为1的业务带宽请求的整体分布
    # App_Priority = [1, 2, 3, 4, 5]
    # ratio_str = 1  # 尽量分离和尽量重用的业务占比
    # Strategy_P = ['Global'] * int(App_num * (1 - ratio_str))
    # Strategy_S = ['Local'] * int(App_num * ratio_str)
    # App_Strategy = Strategy_S + Strategy_P

    import_file = False # 不从excel中读取网络拓扑信息
    Topology = 'Random_SINR'

    # G = Network(Topology, transmit_prob, Coordinates, TX_range, transmit_power, bandwidth, path_loss, noise, import_file)
    # G, Apps = init_func(G, Coordinates, Area_size, CV_range, grid_size, traffic_th, App_num, App_Demand, App_Priority, App_Strategy)
    # 从文件中创建网络和业务对象
    Network_parameters = [Topology, transmit_prob]
    Wireless_parameters = [TX_range, transmit_power, bandwidth]
    Loss_parameters = [path_loss, noise]

    # G_200, Apps_200 = init_function_from_file('Node_Coordinates_200_randomTopo', 'App_100_randomTopo_SLA1_5', Network_parameters, Wireless_parameters, Loss_parameters)
    topology_file = 'Topology_100_Band=10[for_priority_analysis]'
    coordinates_file =  'Node_Coordinates_100_Uniform[for_priority_analysis]'
    app_file = 'App_50_Demand=2_inTopo=100[for_priority_analysis]'
    G, Apps = init_function_from_file(topology_file, coordinates_file, app_file, Network_parameters, Wireless_parameters, Loss_parameters)



    # 业务可用性评估的参数
    T = 30*24
    MTTF, MLife = 2000, 800
    MTTR = 4
    ## 重路由相关的参数
    message_processing_time = 0.05 # 单位为秒s [毫秒量级]
    path_calculating_time = 5 # 单位为秒 s [秒量级]
    detection_rate = 0.99
    demand_th = 1*0.2 # 根据App_demand中的均值来确定
    beta_list = [0.5] # 2类可用性指标的权重(beta越大表明 时间相关的服务可用性水平越重要)
    App_priority_list = [1,2,3,4,5]
    app_priority = App_priority_list * int(len(Apps) / len(App_priority_list))
    random.shuffle(app_priority)
    for i in range(len(Apps)):  # 将业务的优先级设置为 [1~5]
        Apps[i].SLA = app_priority[i]
        # Apps[i].str = 'Global'

    # 业务可用度评估计算
    N = 50 # 网络演化的次数

    # app_results = calculateAvailability(T, G, Apps, MTTF, MLife, MTTR, detection_rate, message_processing_time,  path_calculating_time, beta, demand_th)
    st = time.time()
    # 计算网络拓扑100个节点和200个节点下的业务可用度
    # Multi_app_results_200 = Apps_Availability_MC(N, T, G_200, Apps_200, MTTF, MLife, MTTR, detection_rate, message_processing_time,   path_calculating_time, beta, traffic_th)
    Multi_app_results = Apps_Availability_MC(N, T, G, Apps, MTTF, MLife, MTTR, detection_rate, message_processing_time,   path_calculating_time, beta_list, demand_th)
    # SLA_app_results_200 = calculate_SLA_results(Apps_200, Multi_app_results_200[0])
    SLA_app_results = calculate_SLA_results(Apps, Multi_app_results[0], App_priority_list)
    et = time.time()
    print('\n 采用普通蒙卡计算{}次网络演化的时长为{}s \n'.format(N, et - st))





