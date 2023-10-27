#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   ：Service_availability -> Density_sensitivity_analysis
@IDE    ：PyCharm
@Author ：Yi Zhiwei
@Date   ：2023/7/3 16:51
@Desc   ：网络节点密度的敏感性分析【本质上是链路故障率的敏感性分析】
=================================================='''

import numpy as np
import networkx as nx
import random
import pandas as pd
from Evolution_Model.Evolution_Objects import *
from Evolution_Model.Evolution_Conditions import *
from Evolution_Model.Evolution_Rules import *
from Evaluating_Scripts.Calculating_Availability import *

def save_results(origin_df, file_name):
    # 保存仿真的数据
    # 将dataframe中的数据保存至excel中
    # localtime = time.asctime(time.localtime(time.time()))
    time2 = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M') # 记录数据存储的时间
    sys_path = os.path.abspath('..')  # 表示当前所处文件夹上一级文件夹的绝对路径
    #
    # with pd.ExcelWriter(r'..\Results_saved\{}_time{}.xlsx'.format(file_name, time2)) as xlsx: # 将紧跟with后面的语句求值给as后面的xlsx变量，当with后面的代码块全部被执行完之后，将调用前面返回对象的exit()方法。
    #     origin_df.to_excel(xlsx, sheet_name='app_avail', index=False) # 不显示行索引
    origin_df.to_excel(r'..\Results_Output\LinkFailure_results\{}_{}.xlsx'.format(file_name, time2), index=False)
    print('数据成功保存')

def calculate_SLA_results(Apps, multi_app_res, app_priority_list):
    # 计算不同SLA等级下的业务可用度\业务带宽损失
    App_id_SLA = {} # 存储各优先级下的业务可用度结果
    for pri in app_priority_list:
        App_id_SLA[pri]  = [] # 为每个优先级下的业务创建一个空列表来存储统计的各SLA下业务的可用度
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

def calculate_Density_results(Density_list, N, App_priority_list, beta_list):
    # 服务可用度的网络节点密度敏感性分析[业务仅1个优先级]
    MLife = 800
    SLA_avail = pd.DataFrame(index = App_priority_list)
    WHOLE_avail = pd.DataFrame(index=['一共{}次演化平均'.format(N)])
    EACH_avail = pd.DataFrame(index=list(range(50))) # 假设一共有50个业务
    # path = '\\Different_network_scale\\'

    for density in Density_list:
        print('当前计算的网络拓扑规模为{} \n'.format(density))
        sub_path = '\\Different_network_scale\\Topology_{}[Band=10]+App_50[demand=2]\\'.format(density) # 读取对应的网络和业务excel数据存储的文件夹
        topology_file = sub_path + 'Topology_{}_Band=10'.format(density)
        coordinates_file = sub_path + 'Node_Coordinates_{}_Uniform'.format(density)
        app_file = sub_path + 'App_50_Demand=2_inTopo={}'.format(density)
        G, Apps = init_function_from_file(topology_file, coordinates_file, app_file, Network_parameters, Wireless_parameters, Loss_parameters)

        start_time = time.time()
        current_each_avail, current_whole_avail = Apps_Availability_MC(N, T, G, Apps, MTTF, MLife, MTTR, detection_rate, message_processing_time, path_calculating_time, beta_list, demand_th)
        end_time = time.time()
        print('采用普通蒙卡计算{}次网络演化的时长为{}s \n'.format(N, end_time - start_time))

        current_SLA_avail = calculate_SLA_results(Apps, current_each_avail, App_priority_list)
        current_whole_ave = np.mean(current_whole_avail.iloc[0].tolist())
        SLA_avail.loc[:, density] = pd.Series(current_SLA_avail)  # 每一列存储该MTTF值下的业务可用度
        WHOLE_avail.loc[:, density] = current_whole_ave
        EACH_avail.loc[:, density] = current_each_avail.apply(np.mean, axis=1) # 对每一行求平均值

    return SLA_avail, WHOLE_avail

def priority_analysis_2(TransProb_list, App_priority_list):

    beta_list = [0.5]
    availability_different_priority_local = pd.DataFrame(index=App_priority_list) # 存储结果
    availability_different_priority_global = pd.DataFrame(index=App_priority_list) # 存储结果
    for trans_prob in TransProb_list:
        sub_path = 'Different_link_failure\\Topo_100_transProb={}\\'.format(trans_prob)
        topology_file = sub_path + 'Topology_100_Band=10_transProb={}'.format(trans_prob)
        coordinates_file = 'Different_link_failure\\Node_Coordinates_100_Uniform'
        app_file = sub_path + 'App_30_Demand=2_inTopo={}_Band=10'.format(trans_prob)

        G, Apps = init_function_from_file(topology_file, coordinates_file, app_file, Network_parameters, Wireless_parameters, Loss_parameters)

        app_priority = App_priority_list * int(len(Apps) / len(App_priority_list))  # 乘以每类SLA等级下的业务数量
        random.shuffle(app_priority)
        for i in range(len(Apps)):  # 将业务的优先级设置为 [1~5]
            Apps[i].SLA = app_priority[i]

        # Local 恢复策略下的服务可用度
        start_time = time.time()
        sla_avail, whole_avail = Apps_Availability_MC(N, T, G, Apps, MTTF, MLife, MTTR, detection_rate, message_processing_time, path_calculating_time, beta_list, demand_th)
        end_time = time.time()
        print('采用普通蒙卡计算{}次网络演化的时长为{}s \n'.format(N, end_time - start_time))

        SLA_avail = calculate_SLA_results(Apps, sla_avail, App_priority_list)
        availability_different_priority_local.loc[:, trans_prob] = pd.Series(SLA_avail)  # 每一列存储该transProb值下的业务可用度

        # Global策略下的服务可用度

        for i in range(len(Apps)):  # 将业务的优先级设置为 [1~5]
            Apps[i].str = 'Global'
        start_time_2 = time.time()
        sla_avail, whole_avail = Apps_Availability_MC(N, T, G, Apps, MTTF, MLife, MTTR, detection_rate, message_processing_time, path_calculating_time, beta_list, demand_th)
        end_time_2 = time.time()
        print('采用普通蒙卡计算{}次网络演化的时长为{}s \n'.format(N, end_time_2 - start_time_2))

        SLA_avail = calculate_SLA_results(Apps, sla_avail, App_priority_list)
        availability_different_priority_global.loc[:, trans_prob] = pd.Series(SLA_avail)

    save_results(availability_different_priority_local, 'LinkFailure敏感性分析-不同优先级的服务可用度-{}策略,演化N={}次'.format('Local', N))
    save_results(availability_different_priority_global, 'LinkFailure敏感性分析-不同优先级的服务可用度-{}策略,演化N={}次'.format('Global', N))

    return availability_different_priority_local, availability_different_priority_global

def priority_analysis(Density_list, App_priority_list):
    # 计算不同优先级下的业务可用度
    # N = 20 # 网络演化次数
    # T = 8760 # 网络演化的时长
    beta_list = [0.5]


    availability_different_priority_local = pd.DataFrame(index=App_priority_list) # 存储结果
    availability_different_priority_global = pd.DataFrame(index=App_priority_list) # 存储结果


    for density in Density_list:
        print('当前计算的网络拓扑规模为{} \n'.format(density))
        sub_path = 'Different_network_scale\\Topology_{}[Band=10]+App_30[demand=2]\\'.format(density) # 读取对应的网络和业务excel数据存储的文件夹
        topology_file = sub_path + 'Topology_{}_Band=10'.format(density)
        coordinates_file = sub_path + 'Node_Coordinates_{}_Uniform'.format(density)
        app_file = sub_path + 'App_50_Demand=2_inTopo={}'.format(density)
        G, Apps = init_function_from_file(topology_file, coordinates_file, app_file, Network_parameters, Wireless_parameters, Loss_parameters)

        app_priority = App_priority_list * int(len(Apps) / len(App_priority_list))  # 乘以每类SLA等级下的业务数量
        random.shuffle(app_priority)
        for i in range(len(Apps)):  # 将业务的优先级设置为 [1~5]
            Apps[i].SLA = app_priority[i]

        start_time = time.time()
        sla_avail, whole_avail = Apps_Availability_MC(N, T, G, Apps, MTTF, MLife, MTTR, detection_rate, message_processing_time, path_calculating_time, beta_list, demand_th)
        end_time = time.time()
        print('采用普通蒙卡计算{}次网络演化的时长为{}s \n'.format(N, end_time - start_time))

        SLA_avail = calculate_SLA_results(Apps, sla_avail, App_priority_list)
        availability_different_priority_local.loc[:, density] = pd.Series(SLA_avail)  # 每一列存储该MTTF值下的业务可用度

    save_results(availability_different_priority_local, 'Density敏感性分析-不同优先级的服务可用度-{}策略,演化N={}次'.format('Local', N))
    # draw_line_plot(Density_list, availability_different_priority_local, 'Density敏感性分析-不同优先级的服务可用度-{}策略,演化N={}次'.format('Local', N))


    for density in Density_list:
        print('当前计算的网络拓扑规模为{} \n'.format(density))
        sub_path = '\\Different_network_scale\\Topology_{}[Band=10]+App_50[demand=2]\\'.format(density) # 读取对应的网络和业务excel数据存储的文件夹
        topology_file = sub_path + 'Topology_{}_Band=10'.format(density)
        coordinates_file = sub_path + 'Node_Coordinates_{}_Uniform'.format(density)
        app_file = sub_path + 'App_50_Demand=2_inTopo={}'.format(density)
        G, Apps = init_function_from_file(topology_file, coordinates_file, app_file, Network_parameters, Wireless_parameters, Loss_parameters)

        app_priority = App_priority_list * int(len(Apps) / len(App_priority_list))  # 乘以每类SLA等级下的业务数量
        random.shuffle(app_priority)
        for i in range(len(Apps)):  # 将业务的优先级设置为 [1~5]
            Apps[i].SLA = app_priority[i]
            Apps[i].str = 'Global'

        start_time = time.time()
        sla_avail, whole_avail = Apps_Availability_MC(N, T, G, Apps, MTTF, MLife, MTTR, detection_rate, message_processing_time, path_calculating_time, beta_list, demand_th)
        end_time = time.time()
        print('采用普通蒙卡计算{}次网络演化的时长为{}s \n'.format(N, end_time - start_time))

        SLA_avail = calculate_SLA_results(Apps, sla_avail, App_priority_list)
        availability_different_priority_local.loc[:, density] = pd.Series(SLA_avail)  # 每一列存储该MTTF值下的业务可用度

    save_results(availability_different_priority_local, 'Density敏感性分析-不同优先级的服务可用度-{}策略,演化N={}次'.format('Global', N))
    draw_line_plot(Density_list, availability_different_priority_local, 'Density敏感性分析-不同优先级的服务可用度-{}策略,演化N={}次'.format('Global', N))

    return availability_different_priority_local, availability_different_priority_global

def resource_analysis(TransProb_list):
    # 计算不同传输概率(等价于链路故障率) 下的服务可用度
    beta_list = [0.5]
    App_priority_list = [1]

    availability_different_demand_local = pd.DataFrame(index=[10.2,10.5,20.2,20.5]) # 存储结果
    availability_different_demand_global = pd.DataFrame(index=[10.2,10.5,20.2,20.5]) # 存储结果

    directory_path = 'Different_resourceAndDemand_Topology=100+App=50\\'
    Coordinates_file_name =  directory_path + 'Node_Coordinates_100_Uniform'


    for trans_prob in TransProb_list:
        temp_results = []
        sub_path = 'Different_link_failure\\Topo_100_transProb={}\\'.format(trans_prob)
        coordinates_file = 'Different_link_failure\\Node_Coordinates_100_Uniform'

        file_name_list = [['Topology_100_Band=10_transProb={}'.format(trans_prob), 'App_30_Demand=2_inTopo={}_Band=10'.format(trans_prob)],
                          ['Topology_100_Band=10_transProb={}'.format(trans_prob), 'App_30_Demand=5_inTopo={}_Band=10'.format(trans_prob)],
                          ['Topology_100_Band=20_transProb={}'.format(trans_prob), 'App_30_Demand=2_inTopo={}_Band=20'.format(trans_prob)],
                          ['Topology_100_Band=20_transProb={}'.format(trans_prob), 'App_30_Demand=5_inTopo={}_Band=20'.format(trans_prob)]]  # 待读取的文件列表

        for file_name in file_name_list:
            topology_file = sub_path + file_name[0]
            app_file = sub_path + file_name[1]
            G, Apps = init_function_from_file(topology_file, Coordinates_file_name, app_file,  Network_parameters, Wireless_parameters, Loss_parameters)
            for app_id in range(len(Apps)):
                Apps[app_id].SLA = 1  # 将所有业务等级设置为相同
                # print('业务的优先级为{}'.format(Apps[app_id].SLA))

            sla_avail, whole_avail = Apps_Availability_MC(N, T, G, Apps, MTTF, MLife, MTTR, detection_rate, message_processing_time, path_calculating_time, beta_list, demand_th)
            temp_results.append(whole_avail.apply(np.mean, axis=1).values[0])

        availability_different_demand_local.loc[:,trans_prob] = temp_results
    save_results(availability_different_demand_local, 'LinkFailure敏感性分析-不同资源需求的服务可用度-{}策略,演化N={}次'.format('Local', N))


    for trans_prob in TransProb_list:
        temp_results_glb = []
        sub_path = 'Different_link_failure\\Topo_100_transProb={}\\'.format(trans_prob)
        coordinates_file = 'Different_link_failure\\Node_Coordinates_100_Uniform'

        file_name_list = [['Topology_100_Band=10_transProb={}'.format(trans_prob), 'App_30_Demand=2_inTopo={}_Band=10'.format(trans_prob)],
                          ['Topology_100_Band=10_transProb={}'.format(trans_prob), 'App_30_Demand=5_inTopo={}_Band=10'.format(trans_prob)],
                          ['Topology_100_Band=20_transProb={}'.format(trans_prob), 'App_30_Demand=2_inTopo={}_Band=20'.format(trans_prob)],
                          ['Topology_100_Band=20_transProb={}'.format(trans_prob), 'App_30_Demand=5_inTopo={}_Band=20'.format(trans_prob)]]  # 待读取的文件列表

        for file_name in file_name_list:
            topology_file = sub_path + file_name[0]
            app_file = sub_path + file_name[1]
            G, Apps = init_function_from_file(topology_file, Coordinates_file_name, app_file,  Network_parameters, Wireless_parameters, Loss_parameters)
            for app_id in range(len(Apps)):
                Apps[app_id].SLA = 1  # 将所有业务等级设置为相同
                Apps[app_id].str = 'Global'
                # print('业务的优先级为{}'.format(Apps[app_id].SLA))

            sla_avail_glb, whole_avail_glb = Apps_Availability_MC(N, T, G, Apps, MTTF, MLife, MTTR, detection_rate, message_processing_time, path_calculating_time, beta_list, demand_th)
            temp_results_glb.append(whole_avail_glb.apply(np.mean, axis=1).values[0])

        availability_different_demand_global.loc[:,trans_prob] = temp_results_glb

    save_results(availability_different_demand_global, 'LinkFailure敏感性分析-不同资源需求的服务可用度-{}策略,演化N={}次'.format('Global', N))


    return availability_different_demand_local, availability_different_demand_global





def performance_analysis(Density_list, Beta_list):
    # 计算不同性能比重下的服务可用度
    # N = 50
    # T = 8760
    availability_different_beta_local = pd.DataFrame(index = Beta_list)
    availability_different_beta_global = pd.DataFrame(index = Beta_list)


    for density in Density_list:
        print('当前计算的Density规模值为{} \n'.format(density))
        sub_path = '\\Different_network_scale\\Topology_{}[Band=10]+App_50[demand=2]\\'.format(density)  # 读取对应的网络和业务excel数据存储的文件夹
        topology_file = sub_path + 'Topology_{}_Band=10'.format(density)
        coordinates_file = sub_path + 'Node_Coordinates_{}_Uniform'.format(density)
        app_file = sub_path + 'App_50_Demand=2_inTopo={}'.format(density)
        G, Apps = init_function_from_file(topology_file, coordinates_file, app_file, Network_parameters,
                                          Wireless_parameters, Loss_parameters)
        for app_id in range(len(Apps)):  # 将业务优先级统一为1
            Apps[app_id].SLA = 1

        start_time = time.time()
        multi_beta_avail = pd.DataFrame(index= Beta_list)

        for n in range(N):
            st_time = time.time()
            G_tmp = copy.deepcopy(G)
            App_tmp = copy.deepcopy(Apps)
            SLA_avail, whole_avail  = calculateAvailability(T, G_tmp, App_tmp, MTTF, MLife, MTTR, detection_rate, message_processing_time, path_calculating_time, Beta_list, demand_th)
            # print('当前第{}次循环业务的可用度为{}'.format(n, result[0][1]))
            multi_beta_avail.loc[:, n + 1] = whole_avail  # 将单次演化下各业务的可用度结果存储为dataframe中的某一列(index为app_id)，其中n+1表示列的索引
            ed_time = time.time()
            print('\n 当前为第{}次蒙卡仿真, 仿真时长为{}s'.format(n, ed_time - st_time))

        availability_different_beta_local.loc[:, density] = multi_beta_avail.apply(np.mean, axis=1) # 对每行[各次蒙卡]下的整网可用度求平均值；apply function to each row.

        end_time = time.time()
        print('采用普通蒙卡计算{}次网络演化的时长为{}s \n'.format(N, end_time-start_time))

    save_results(availability_different_beta_local, 'Density敏感性分析-不同性能权重的服务可用度-{}策略,演化N={}次'.format('Local', N))
    # draw_line_plot(Density_list, availability_different_beta_local, 'Density敏感性分析-不同性能权重的服务可用度-{}策略,演化N={}次'.format('Local', N))

    for density in Density_list:
        print('当前计算的Density规模值为{} \n'.format(density))
        sub_path = '\\Different_network_scale\\Topology_{}[Band=10]+App_50[demand=2]\\'.format(density)  # 读取对应的网络和业务excel数据存储的文件夹
        topology_file = sub_path + 'Topology_{}_Band=10'.format(density)
        coordinates_file = sub_path + 'Node_Coordinates_{}_Uniform'.format(density)
        app_file = sub_path + 'App_50_Demand=2_inTopo={}'.format(density)
        G, Apps = init_function_from_file(topology_file, coordinates_file, app_file, Network_parameters,
                                          Wireless_parameters, Loss_parameters)
        for app_id in range(len(Apps)):  # 将业务优先级统一为1
            Apps[app_id].SLA = 1
            Apps[app_id].str = 'Global'

        start_time = time.time()
        multi_beta_avail = pd.DataFrame(index= Beta_list)

        for n in range(N):
            st_time = time.time()
            G_tmp = copy.deepcopy(G)
            App_tmp = copy.deepcopy(Apps)
            SLA_avail, whole_avail  = calculateAvailability(T, G_tmp, App_tmp, MTTF, MLife, MTTR, detection_rate, message_processing_time, path_calculating_time, Beta_list, demand_th)
            # print('当前第{}次循环业务的可用度为{}'.format(n, result[0][1]))
            multi_beta_avail.loc[:, n + 1] = whole_avail  # 将单次演化下各业务的可用度结果存储为dataframe中的某一列(index为app_id)，其中n+1表示列的索引
            ed_time = time.time()
            print('\n 当前为第{}次蒙卡仿真, 仿真时长为{}s'.format(n, ed_time - st_time))

        availability_different_beta_local.loc[:, density] = multi_beta_avail.apply(np.mean, axis=1) # 对每行[各次蒙卡]下的整网可用度求平均值；apply function to each row.

        end_time = time.time()
        print('采用普通蒙卡计算{}次网络演化的时长为{}s \n'.format(N, end_time-start_time))


    save_results(availability_different_beta_global, 'Density敏感性分析-不同性能权重的服务可用度-{}策略,演化N={}次'.format('Global', N))

    # draw_line_plot(Density_list, availability_different_beta_local, 'Density敏感性分析-不同性能权重的服务可用度-{}策略,演化N={}次'.format('Local', N))
    # draw_line_plot(Density_list, availability_different_beta_global, 'Density敏感性分析-不同性能权重的服务可用度-{}策略,演化N={}次'.format('Global', N))


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
    ax1.set_xlabel('Number of nodes')
    ax1.set_ylabel('Service Availability')
    # plt.legend(title="Priority")
    plt.savefig(r'..\Pictures_Saved\折线图{}.jpg'.format('{}'.format(file_name), dpi=1200) )
    plt.show()

if __name__ == '__main__':

    ## 无线传输相关的参数
    transmit_prob = 0.1  # 节点的传输概率
    transmit_power = 1.5  # 发射功率(毫瓦)，统一单位：W
    path_loss = 2  # 单位：无
    noise = pow(10,
                -11)  # 噪声的功率谱密度(毫瓦/赫兹)，统一单位：W/Hz, 参考自https://dsp.stackexchange.com/questions/13127/snr-calculation-with-noise-spectral-density
    bandwidth = 10 * pow(10, 6)  # 带宽(Mhz)，统一单位：Hz
    lambda_TH = 8 * pow(10, -1)  # 接收器的敏感性阈值,用于确定节点的传输范围
    TX_range = 30
    CV_range = 30  # 节点的覆盖范围
    Topology = 'Random_SINR'

    ## 1. 业务的优先级分析
    App_priority_list = [1, 2, 3, 4, 5]
    topology_file = 'Topology_100_Band=10[for_priority_analysis]'
    coordinates_file = 'Node_Coordinates_100_Uniform[for_priority_analysis]'
    app_file = 'App_50_Demand=2_inTopo=100[for_priority_analysis]'

    Network_parameters = [Topology, transmit_prob]
    Wireless_parameters = [TX_range, transmit_power, bandwidth]
    Loss_parameters = [path_loss, noise]

    ## 服务可用性评估相关的参数
    N = 5
    T = 876
    message_processing_time = 0.05  # 单位为秒 s
    path_calculating_time = 5  # 单位为秒 s
    detection_rate = 0.99
    demand_th =  2*math.pow((1/1),1)*math.exp(-1)  # 根据App_demand中的均值来确定
    beta_list = [0.5]  # 2类可用性指标的权重(beta越大表明 时间相关的服务可用性水平越重要)

    MTTF = 2000
    MLife = 800
    MTTR = 4

    # Density_list = [ 80, 90, 100, 110, 120, 130, 140, 150] # 一共8组网络规模的数据
    TransProb_list = [0.01,0.02,0.03,0.04,0.05] #,0.06,0.07,0.08,0.09,0.1,0.11,0.12,0.13,0.14,0.15,0.16,0.17,0.18,0.19,0.2] #
    local_res, global_res = resource_analysis(TransProb_list)

    print('==================')
    print('资源敏感性的服务可用度评估已经计算完成....\n')

    # loc_res, glb_res = priority_analysis_2(TransProb_list, App_priority_list)
    # print('服务优先级下的可用度评估已经计算完成....\n')



    # local_res, global_res = priority_analysis(Density_list, App_priority_list)
    #
    # Beta_list = [0.1, 0.3, 0.5, 0.7, 0.9]
    # local_pf, global_pf = performance_analysis(Density_list, Beta_list)