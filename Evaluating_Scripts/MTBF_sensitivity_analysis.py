#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   ：Service_availability -> MTBF_sensitivity_analysis
@IDE    ：PyCharm
@Author ：Yi Zhiwei
@Date   ：2023/4/5 19:28
@Desc   ：不同构件MTTF值下业务可用度结果的敏感性分析
=================================================='''

import numpy as np
import networkx as nx
import random
import pandas as pd

from Evaluating_Scripts.Calculating_Availability import *
from Evolution_Model.Evolution_Objects import *
from Evolution_Model.Evolution_Conditions import *
from Evolution_Model.Evolution_Rules import *
from Evaluating_Scripts.Calculating_Availability import *

def save_results(origin_df, file_name):
    # 保存仿真的数据
    # 将dataframe中的数据保存至excel中
    # localtime = time.asctime(time.localtime(time.time()))
    time2 = datetime.datetime.now().strftime('%Y_%m_%d_+%H_%M') # 记录数据存储的时间
    sys_path = os.path.abspath('..')  # 表示当前所处文件夹上一级文件夹的绝对路径
    #
    # with pd.ExcelWriter(r'..\Results_saved\{}_time{}.xlsx'.format(file_name, time2)) as xlsx: # 将紧跟with后面的语句求值给as后面的xlsx变量，当with后面的代码块全部被执行完之后，将调用前面返回对象的exit()方法。
    #     origin_df.to_excel(xlsx, sheet_name='app_avail', index=False) # 不显示行索引
    origin_df.to_excel(r'..\Results_Output\MTTF_results\{}_{}.xlsx'.format(file_name, time2), index=False)
    print('数据成功保存')

def calculate_SLA_results(Apps, multi_app_res, app_priority_list):
    # 计算不同SLA等级下的业务可用度\业务带宽损失
    App_id_SLA = {}  # 存储各优先级下的业务可用度结果
    for pri in app_priority_list:
        App_id_SLA[pri] = []  # 为每个优先级下的业务创建一个空列表来存储统计的各SLA下业务的可用度
    ## 1.先统计不同SLA对应的业务id
    for i in range(len(Apps)):
        SLA = Apps[i].SLA
        App_id_SLA[SLA].append(i)

    ## 2.再计算不同SLA下各业务的平均值( average=(该SLA下单个业务的N次演化平均)/Num_SLA )
    SLA_res = {}
    for SLA, app_list in App_id_SLA.items():
        Ave_SLA = [] # 统计各SLA下业务多次演化的平均值
        for app_id in app_list:
            res = np.mean(multi_app_res.loc[app_id])
            Ave_SLA.append(res)
        SLA_res[SLA] = np.mean(Ave_SLA) # 对SLA等级下的所有业务可用度的均值再求均值，作为该等级下的业务可用度
    return SLA_res


def calculate_MTTF_analysis(MTTF_list, N, G, Apps, App_priority_list, beta_list):
    # 业务可用性的MTTF敏感性分析
    MLife = 800
    SLA_avail = pd.DataFrame(index=App_priority_list) # 存储业务可用度数据的行索引为各SLA的值
    WHOLE_avail = pd.DataFrame(index=['一共{}次演化平均'.format(N)]) # 存储业务可用度数据的行索引为演化次数的值
    EACH_avail = pd.DataFrame(index=list(Apps.keys()))


    for mttf in MTTF_list:
        print('当前计算的MTTF值为{} \n'.format(mttf))
        start_time = time.time()
        current_each_avail, current_whole_avail = Apps_Availability_MC(N, T,  G, Apps, mttf, MLife, MTTR, detection_rate, message_processing_time, path_calculating_time, beta_list, demand_th)
        end_time = time.time()
        print('采用普通蒙卡计算{}次网络演化的时长为{}s \n'.format(N, end_time-start_time))

        current_SLA_avail = calculate_SLA_results(Apps, current_each_avail, App_priority_list) # 当前N次演化下各SLA等级的服务可用度
        # whole_avail = res[1].apply(np.mean, axis=1) # 对dataframe中的每一行应用求平均值
        current_whole_ave = np.mean(current_whole_avail.iloc[0].tolist()) #当前N次演化下整网服务可用度求均值
        SLA_avail.loc[:, mttf] = pd.Series(current_SLA_avail) # 每一列存储该MTTF值下的业务可用度
        WHOLE_avail.loc[:, mttf] = current_whole_ave
        EACH_avail.loc[:, mttf] = current_each_avail.apply(np.mean, axis=1) # 对每一行求平均值


    return SLA_avail, WHOLE_avail

def priority_analysis(MTTF_list, App_priority_list, G, Apps):
    # 计算不同优先级下的业务可用度
    # N = 20 # 网络演化次数
    # T = 8760 # 网络演化的时长
    beta_list = [0.5]
    app_priority = App_priority_list * int(len(Apps) / len(App_priority_list))  # 乘以每类SLA等级下的业务数量
    random.shuffle(app_priority)
    for i in range(len(Apps)):  # 将业务的优先级设置为 [1~5]
        Apps[i].SLA = app_priority[i]

    availability_different_priority_local = pd.DataFrame(index=App_priority_list) # 存储结果
    availability_different_priority_global = pd.DataFrame(index=App_priority_list) # 存储结果


    for mttf in MTTF_list:
        print('当前计算的MTTF值为{} \n'.format(mttf))
        start_time = time.time()
        SLA_avail, whole_avail = Apps_Availability_MC(N, T,  G, Apps, mttf, MLife, MTTR, detection_rate, message_processing_time, path_calculating_time, beta_list, demand_th)
        end_time = time.time()
        print('采用普通蒙卡计算{}次网络演化的时长为{}s \n'.format(N, end_time-start_time))

        SLA_avail = calculate_SLA_results(Apps, SLA_avail, App_priority_list)
        availability_different_priority_local.loc[:, mttf] = pd.Series(SLA_avail) # 每一列存储该MTTF值下的业务可用度

    save_results(availability_different_priority_local, 'MTTF敏感性分析-不同优先级的服务可用度-{}策略,演化N={}次,{}节点的拓扑'.format(Apps[0].str, N, len(G)))
    draw_line_plot(MTTF_list, availability_different_priority_local, 'MTTF敏感性分析-不同优先级的服务可用度-{}策略,演化N={}次,{}节点的拓扑'.format(Apps[0].str, N, len(G)) )

    for i in range(len(Apps)):  # 将业务的优先级设置为 [1~5]
        Apps[i].str = 'Global'  # 将业务的策略设置为Global

    for mttf in MTTF_list:

        print('当前计算的MTTF值为{} \n'.format(mttf))
        start_time = time.time()
        SLA_avail, whole_avail = Apps_Availability_MC(N, T,  G, Apps, mttf, MLife, MTTR, detection_rate, message_processing_time, path_calculating_time, beta_list, demand_th)
        end_time = time.time()
        print('采用普通蒙卡计算{}次网络演化的时长为{}s \n'.format(N, end_time-start_time))

        SLA_avail = calculate_SLA_results(Apps, SLA_avail, App_priority_list)
        availability_different_priority_global.loc[:, mttf] = pd.Series(SLA_avail) # 每一列存储该MTTF值下的业务可用度

    save_results(availability_different_priority_global, 'MTTF敏感性分析-不同优先级的服务可用度-{}策略,演化N={}次,{}节点的拓扑'.format(Apps[0].str, N, len(G)))
    draw_line_plot(MTTF_list, availability_different_priority_global, 'MTTF敏感性分析-不同优先级的服务可用度-{}策略,演化N={}次,{}节点的拓扑'.format(Apps[0].str, N, len(G)) )

    return availability_different_priority_local, availability_different_priority_global

def resource_analysis(MTTF_list, File_name_list):
    # 计算不同网络带宽和业务请求下的业务可用度
    # N = 50
    # T = 8760

    beta_list = [0.5]
    App_priority_list = [1]

    directory_path = 'Different_resourceAndDemand_Topology=100+App=50\\'
    Coordinates_file_name =  directory_path + 'Node_Coordinates_100_Uniform'
    file_name_list = [['Topology_100_Band=10', 'App_50_Demand=2_inTopo=100_Band=10'], ['Topology_100_Band=10', 'App_50_Demand=5_inTopo=100_Band=10'],
                      ['Topology_100_Band=20', 'App_50_Demand=2_inTopo=100_Band=20'],
                      ['Topology_100_Band=20', 'App_50_Demand=5_inTopo=100_Band=20']]  # 待读取的文件列表

    availability_different_demand_local = pd.DataFrame(index=MTTF_list) # 存储结果
    availability_different_demand_global = pd.DataFrame(index=MTTF_list)


    for file_name in file_name_list:
        print('当前计算的网络和业务规模为{} \n'.format(file_name))
        topology_file = directory_path + file_name[0]
        app_file = directory_path + file_name[1]
        G, Apps = init_function_from_file(topology_file, Coordinates_file_name, app_file,  Network_parameters, Wireless_parameters, Loss_parameters)
        for app_id in range(len(Apps)):
            Apps[app_id].SLA = 1 # 将所有业务等级设置为相同
            print('业务的优先级为{}'.format(Apps[app_id].SLA))

        t1 = time.time()
        sla_avail_1, whole_avail_1 = calculate_MTTF_analysis(MTTF_list, N, G, Apps, App_priority_list, beta_list)
        save_results(whole_avail_1, 'MTTF敏感性分析,网络规模[{}]-整网平均-{}策略,演化N={}次,{}节点的拓扑'.format(file_name[0]+file_name[1], Apps[0].str, N, len(G)))

        # availability_different_demand_local.loc[ :, file_name] = whole_avail_1.T # 每一列存储各文件对应的整网服务可用度
        t2 = time.time()
        print('\n 当前{}策略计算的总时长为{}h'.format(Apps[0].str, (t2 - t1) / 3600))



        # 将业务的策略修改为Global
        for app_id in range(len(Apps)):
            Apps[app_id].str = 'Global'

        t3 = time.time()
        sla_avail_2, whole_avail_2 = calculate_MTTF_analysis(MTTF_list, N, G, Apps, App_priority_list, beta_list)
        save_results(whole_avail_2, 'MTTF敏感性分析,网络规模[{}]-整网平均-{}策略,演化N={}次,{}节点的拓扑'.format(file_name[0]+file_name[1], Apps[0].str, N, len(G)))
        # availability_different_demand_global.loc[:, file_name] = whole_avail_2.T

        t4 = time.time()
        print('\n 当前{}策略计算的总时长为{}h'.format(Apps[0].str, (t3 - t4) / 3600))

    return availability_different_demand_local, availability_different_demand_global


def performance_analysis(MTTF_list, Beta_list, G, Apps):
    # 计算不同性能比重下的服务可用度
    # N = 20
    # T = 8760
    availability_different_beta_local = pd.DataFrame(index = Beta_list)
    availability_different_beta_global = pd.DataFrame(index = Beta_list)


    for app_id in range(len(Apps)): # 将业务优先级统一为1
        Apps[app_id].SLA = 1

    for mttf in MTTF_list:
        print('当前计算的MTTR值为{} \n'.format(mttf))
        start_time = time.time()
        multi_beta_avail = pd.DataFrame(index= Beta_list)

        for n in range(N):
            st_time = time.time()
            G_tmp = copy.deepcopy(G)
            App_tmp = copy.deepcopy(Apps)
            print('业务5的初始路径为{}'.format(App_tmp[5].path))
            SLA_avail, whole_avail  = calculateAvailability(T, G_tmp, App_tmp, mttf, MLife, MTTR, detection_rate,
                                           message_processing_time, path_calculating_time, Beta_list, demand_th)
            multi_beta_avail.loc[:, n + 1] = whole_avail  # 将单次演化下各业务的可用度结果存储为dataframe中的某一列(index为app_id)，其中n+1表示列的索引
            ed_time = time.time()
            print('\n 当前为第{}次蒙卡仿真, 仿真时长为{}s'.format(n, ed_time - st_time))

        availability_different_beta_local.loc[:, mttf] = multi_beta_avail.apply(np.mean, axis=1) # 对每行[各次蒙卡]下的整网可用度求平均值；apply function to each row.

        end_time = time.time()
        print('采用普通蒙卡计算{}次网络演化的时长为{}s \n'.format(N, end_time-start_time))

    save_results(availability_different_beta_local, 'MTTF敏感性分析-不同性能权重的服务可用度-{}策略,演化N={}次,{}节点的拓扑'.format(Apps[0].str, N, len(G)))

    for app_id in range(len(Apps)): # 将业务策略设置为GLobal
        Apps[app_id].str = 'Global'

    for mttf in MTTF_list:
        print('当前计算的MTTF值为{} \n'.format(mttf))
        start_time = time.time()
        multi_beta_avail = pd.DataFrame(index= Beta_list)

        for n in range(N):
            st_time = time.time()
            G_tmp = copy.deepcopy(G)
            App_tmp = copy.deepcopy(Apps)
            SLA_avail, whole_avail  = calculateAvailability(T, G_tmp, App_tmp, mttf, MLife, MTTR, detection_rate,
                                           message_processing_time, path_calculating_time, Beta_list, demand_th)
            multi_beta_avail.loc[:, n + 1] = whole_avail  # 将单次演化下各业务的可用度结果存储为dataframe中的某一列(index为app_id)，其中n+1表示列的索引
            ed_time = time.time()
            print('\n 当前为第{}次蒙卡仿真, 仿真时长为{}s'.format(n, ed_time - st_time))

        availability_different_beta_global.loc[:, mttf] = multi_beta_avail.apply(np.mean, axis=1) # 对每行[各次蒙卡]下的整网可用度求平均值；apply function to each row.

        end_time = time.time()
        print('采用普通蒙卡计算{}次网络演化的时长为{}s \n'.format(N, end_time-start_time))

    save_results(availability_different_beta_global, 'MTTF敏感性分析-不同性能权重的服务可用度-{}策略,演化N={}次,{}节点的拓扑'.format(Apps[0].str, N, len(G)))

    draw_line_plot(MTTF_list, availability_different_beta_local, 'MTTF敏感性分析-不同性能权重的服务可用度-{}策略,演化N={}次,{}节点的拓扑'.format(Apps[0].str, N, len(G)) )
    draw_line_plot(MTTF_list, availability_different_beta_global, 'MTTF敏感性分析-不同性能权重的服务可用度-{}策略,演化N={}次,{}节点的拓扑'.format(Apps[0].str, N, len(G)) )

    return availability_different_beta_local, availability_different_beta_global

def draw_line_plot(x_data, y_data, file_name):
    time2 = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M') # 记录数据存储的时间

    x_data = x_data
    y_data = y_data# 不同SLA下的业务可用度

    fig1, ax1 = plt.subplots()
    fig1.subplots_adjust(hspace=0.5) # make a little extra space between the subplots
    # colors = ['gold','blue','green'] # ,'orangered','hotpink'
    if len(y_data.index) > 1: # 如果纵坐标有多行数据
        for i in y_data.index:
            ax1.plot(x_data, y_data.loc[i], label='${}$'.format(i)) # i+1表示业务等级 c=colors[i]
    else:
        ax1.plot(x_data, y_data)
    ax1.set_xlabel('MTTF of components')
    ax1.set_ylabel('Service Availability')
    # plt.legend(title="Priority")
    plt.savefig(r'..\Pictures_Saved\折线图{}.jpg'.format('{}'.format(file_name), dpi=1200) )
    plt.show()


if __name__ == '__main__':
    # 网络演化对象的输入参数；
    # 生成简单的case测试各function是否能正常输出结果
    ## 无线传输相关的参数
    transmit_prob = 0.1 # 节点的传输概率
    transmit_power = 1.5  # 发射功率(毫瓦)，统一单位：W
    path_loss = 2  # 单位：无
    noise = pow(10, -11)  # 噪声的功率谱密度(毫瓦/赫兹)，统一单位：W/Hz, 参考自https://dsp.stackexchange.com/questions/13127/snr-calculation-with-noise-spectral-density
    bandwidth = 10 * pow(10, 6)  # 带宽(Mhz)，统一单位：Hz
    lambda_TH = 8 * pow(10, -1)  # 接收器的敏感性阈值,用于确定节点的传输范围
    TX_range = 30
    CV_range = 30  # 节点的覆盖范围
    Topology = 'Random_SINR'


    ## 1. 业务的优先级分析
    App_priority_list = [1, 2, 3, 4, 5]
    topology_file = 'Topology_100_Band=10[for_priority_analysis]'
    coordinates_file =  'Node_Coordinates_100_Uniform[for_priority_analysis]'
    app_file = 'App_50_Demand=2_inTopo=100[for_priority_analysis]' # 'App_50_Demand=2_inTopo=100_Band=10' #

    Network_parameters = [Topology, transmit_prob]
    Wireless_parameters = [TX_range, transmit_power, bandwidth]
    Loss_parameters = [path_loss, noise]


    ## 服务可用性评估相关的参数
    N = 50
    # T = 30 * 24 # 一个月
    T = 8760 # 一年时长
    message_processing_time = 0.05 # 单位为秒 50ms
    path_calculating_time = 5 # 单位为秒 s
    detection_rate = 0.99
    demand_th = 0.2 # 根据App_demand中的均值来确定
    beta_list = [0.5] # 2类可用性指标的权重(beta越大表明 时间相关的服务可用性水平越重要)

    # MTTF = 2000
    MTTR = 4
    MLife = 800
    MTTF_list = np.linspace(1000, 2000, 41) # 40个点


    G, Apps = init_function_from_file(topology_file, coordinates_file, app_file, Network_parameters, Wireless_parameters, Loss_parameters)



    local_res, global_res = priority_analysis(MTTF_list, App_priority_list, G, Apps)

    File_name_list = ['暂无，从函数中内置了待读取的文件列表']
    local_, global_ = resource_analysis(MTTF_list, File_name_list)

    Beta_list = [0.1, 0.3, 0.5, 0.7, 0.9]
    local_pf, global_pf = performance_analysis(MTTF_list, Beta_list, G, Apps)


    # 对计算结果进行图形化的展示
    # time2 = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M') # 记录数据存储的时间
    #
    # x_data = MTTF_list
    # y_data = Res[0] # 不同SLA下的业务可用度
    # y2_data = Res[1]
    #
    # fig1, ax1 = plt.subplots()
    # fig1.subplots_adjust(hspace=0.5) # make a little extra space between the subplots
    # colors = ['gold','blue','green'] # ,'orangered','hotpink'
    # for i in range(len(colors)):
    #     ax1.plot(x_data, y_data.loc[i+1], c=colors[i], label='${}$'.format(i+1)) # i+1表示业务等级
    # # ax1.plot(x_data, y2_data.iloc[0], label='whole network')
    # ax1.set_xlabel('MTTF of components')
    # ax1.set_ylabel('Service Availability')
    # plt.legend(loc="upper right", title="Priority")
    # plt.savefig(r'..\Pictures_Saved\line_plot_{}.jpg'.format('MTTF敏感性分析[Band=20,Demand=5]-Local演化N={}次-节点数量为{}'.format(N, len(G)), dpi=1200))
    #
    # plt.show()
    #
    # save_results(Res[0], 'MTTF敏感性分析[Band=20,Demand=5]-Local演化N=10次,100节点的拓扑')



