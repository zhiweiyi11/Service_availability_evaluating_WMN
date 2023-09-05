from Evolution_Model.Evolution_Objects import *
from Evolution_Model.Evolution_Conditions import *
from Evolution_Model.Evolution_Rules import *
from Evolution_Model.Application_request_generating import *
from Evaluating_Scripts.Calculating_Availability import *
from Evaluating_Scripts.MTBF_sensitivity_analysis import *
from concurrent.futures import ThreadPoolExecutor

import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    # 网络演化对象的输入参数；
    # 网络演化对象的输入参数；
    save_data = False  # 是否保存节点坐标数据
    Node_num = 100
    Area_size = (150, 150)
    Area_width, Area_length = 150, 150
    Coordinates = generate_positions(Node_num, Area_width, Area_length, save_data)

    # TX_range = 50 # 传输范围为区域面积的1/5时能够保证网络全联通
    transmit_prob = 0.1  # 节点的传输概率
    transmit_power = 1.5  # 发射功率(毫瓦)，统一单位：W
    path_loss = 2  # 单位：无
    noise = pow(10, -11)  # 噪声的功率谱密度(毫瓦/赫兹)，统一单位：W/Hz, 参考自https://dsp.stackexchange.com/questions/13127/snr-calculation-with-noise-spectral-density
    bandwidth = 10 * pow(10, 6)  # 带宽(Mhz)，统一单位：Hz
    lambda_TH = 8 * pow(10, -1)  # 接收器的敏感性阈值,用于确定节点的传输范围
    # TX_range = pow((transmit_power / (bandwidth * noise * lambda_TH)), 1 / path_loss)
    TX_range = 30
    CV_range = 30  # 节点的覆盖范围

    import_file = False  # 不从excel中读取网络拓扑信息
    Topology = 'Random_SINR'
    # 从文件中创建网络和业务对象
    Network_parameters = [Topology, transmit_prob]
    Wireless_parameters = [TX_range, transmit_power, bandwidth]
    Loss_parameters = [path_loss, noise]

    topology_file_star = '../Results_Saving/Small_Scale_Network/topo_star'
    topology_file_P2P = '../Results_Saving/Small_Scale_Network/topo_P2P'
    topology_file_tree = '../Results_Saving/Small_Scale_Network/topo_tree'
    topology_file_mesh = '../Results_Saving/Small_Scale_Network/topo_mesh'

    # coordinates_file 暂时没用到
    coordinates_file = 'Node_Coordinates_100_Uniform[for_priority_analysis]'

    app_file_star = '../Results_Saving/Small_Scale_Network/App_3_star'
    app_file_P2P = '../Results_Saving/Small_Scale_Network/App_3_P2P'
    app_file_tree = '../Results_Saving/Small_Scale_Network/App_3_tree'
    app_file_mesh = '../Results_Saving/Small_Scale_Network/App_2_mesh'

    # G_star, Apps_star = init_function_from_file(topology_file_star, coordinates_file, app_file_star, Network_parameters,
    #                                           Wireless_parameters, Loss_parameters)
    # G_P2P, Apps_P2P = init_function_from_file(topology_file_P2P, coordinates_file, app_file_P2P, Network_parameters,
    #                                           Wireless_parameters, Loss_parameters)
    # G_tree, Apps_tree = init_function_from_file(topology_file_tree, coordinates_file, app_file_tree, Network_parameters,
    #                                           Wireless_parameters, Loss_parameters)
    G_mesh, Apps_mesh = init_function_from_file(topology_file_mesh, coordinates_file, app_file_mesh, Network_parameters,
                                              Wireless_parameters, Loss_parameters)

    # 业务可用性评估的参数
    T = 8760
    MTTF, MLife = 2000, 800
    MTTR = 4
    ## 重路由相关的参数
    message_processing_time = 0.05#.05  # 单位为秒s [毫秒量级]
    path_calculating_time = 0.5 #.5  # 单位为秒 s [秒量级]
    detection_rate = 1
    demand_th = 2*math.pow((1/1),1)*math.exp(-1)  # 根据App_demand中的均值来确定,表示性能损失占业务带宽需求的比例
    beta_list = [0.5]  # 2类可用性指标的权重(beta越大表明 时间相关的服务可用性水平越重要)
    App_priority_list = [1, 2, 3, 4, 5]
    # app_priority = App_priority_list * int(len(Apps) / len(App_priority_list))
    # random.shuffle(app_priority)
    # for i in range(len(Apps)):  # 将业务的优先级设置为 [1~5]
    #     Apps[i].SLA = app_priority[i]
    nx.draw_networkx(G_mesh, pos=nx.spring_layout(G_mesh))
    plt.show()
    # 收敛性分析的参数
    N = 200
    for i in range(len(Apps_mesh)):  # 将业务的优先级设置为 [1~5]
        # Apps[i].SLA = app_priority[i]
        Apps_mesh[i].str = 'Local'
    # single_results_star, whole_results_star = Apps_Availability_MC(N, T, G_star, Apps_star, MTTF, MLife, MTTR,
    #                                                              detection_rate,
    #                                                              message_processing_time, path_calculating_time,
    #                                                              beta_list,
    #                                                              demand_th)

    single_results_mesh, whole_results_mesh = Apps_Availability_MC(N, T, G_mesh, Apps_mesh, MTTF, MLife, MTTR,
                                                                 detection_rate,   message_processing_time, path_calculating_time,
                                                                 beta_list, demand_th)

    # single_results_tree, whole_results_tree = Apps_Availability_MC(N, T, G_tree, Apps_tree, MTTF, MLife, MTTR,
    #                                                              detection_rate,
    #                                                              message_processing_time, path_calculating_time,
    #                                                              beta_list,
    #                                                              demand_th)

    # print(single_results_star)
    print(single_results_mesh)
    # print(single_results_tree)

    temp1 = single_results_mesh.to_numpy()
    availability_mesh_app_1 = temp1[0, :]
    availability_mesh_app_2 = temp1[1, :]

    temp2 = whole_results_mesh.to_numpy()
    availability_P2P_whole = temp2[0, :]

    # bins = np.linspace(0.9, 1, 21)

    fig, ax = plt.subplots()
    n1, bins1, patches = ax.hist(availability_mesh_app_1, bins=20)
    n2, bins2, patches2 = ax.hist(availability_P2P_whole, bins=20)

    ax.plot(bins1[:20], n1, marker='o', color='red', linestyle='--')
    ax.plot(bins2[:20], n2, marker='o', color='green', linestyle='--')


    # n, bins, patches = plt.hist(availability_P2P_app_1, bins=30, density=True, stacked=True)
    plt.show(block=True)
