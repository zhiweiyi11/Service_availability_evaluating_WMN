from Evolution_Model.Evolution_Objects import *
from Evolution_Model.Evolution_Conditions import *
from Evolution_Model.Evolution_Rules import *
from Evolution_Model.Application_request_generating import *
from Evaluating_Scripts.Calculating_Availability import *
from Evaluating_Scripts.MTBF_sensitivity_analysis import *
from concurrent.futures import ThreadPoolExecutor

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import statsmodels.api as sm
import pylab
# 调用Reliability的包
from reliability.Probability_plotting import QQ_plot_semiparametric, PP_plot_semiparametric
from reliability.Fitters import Fit_Weibull_2P, Fit_Normal_2P
from reliability.Distributions import Normal_Distribution



def draw_QQ_plot(measured_data):
    app1_data = measured_data[0]
    app2_data = measured_data[1]

    dist_1 = Fit_Normal_2P(failures=app1_data, print_results=True, show_probability_plot=True).distribution  # fit a normal distribution
    dist_2 = Fit_Normal_2P(failures=app2_data, print_results=True, show_probability_plot=True).distribution

    plt.figure(figsize=(10, 5))
    # plt.subplot(121)
    stats.probplot(app1_data, dist="norm", plot=pylab, fit=True)
    # PP_plot_semiparametric(X_data_failures=app1_data, Y_dist=dist_1)
    # QQ_plot_semiparametric(X_data_failures=app1_data, Y_dist=dist_1, show_fitted_lines=False, show_diagonal_line=True)
    # plt.title('app $1$')
    # plt.subplot(122)
    # QQ_plot_semiparametric(X_data_failures=app1_data, Y_dist=dist_2, show_fitted_lines=False, show_diagonal_line=True)
    # PP_plot_semiparametric(X_data_failures=app2_data, Y_dist=dist_2)
    stats.probplot(app2_data, dist='norm', plot=pylab, fit=True)
    # plt.title('app $2$')
    pylab.show()

def Mesh_RBD_(MTTF, MTTR, link_fail):
    A1, A2 = 0, 0
    a = MTTF/(MTTF+MTTR)
    a1_shared = 1 - ((1-a)*(1-a)+ (1-a)*a*(link_fail + link_fail*link_fail))
    A1 = a * a1_shared * a1_shared
    a2_shared = 1- ((1-a)*(1-a)+ (1-a)*a*(link_fail + link_fail*link_fail)) # 用"1-不可用度"来表示节点3和4互为备份时的可用度
    A2 = a * a* a2_shared

    return A1, A2

def Tree_RBD(MTTF, MTTR, link_fail):
    A1, A2 = 0, 0

    a = MTTF/(MTTF+MTTR)
    a1_shared = 1 - ((1-a)*(1-a) + (1-a) * a * (1-pow(1-link_fail,3)*(1-link_fail*link_fail)) ) # 其中新路径的路由不可用度为 1-pow(1-link_fail,3)*(1-link_fail*link_fail)
    a2_shared = 1- ((1-a)*(1-a) +  (1-a) * a * (2*link_fail + link_fail*link_fail))
    # a2_shared = 1 -  4 * (1-a)*a*a*a*(2*link_fail + link_fail*link_fail) - 2 * (1-a)*(1-a)*a*a*(2*link_fail + link_fail*link_fail) - 2 * (1-a)*(1-a)*a*a # 把所有的故障场景都枚举出来了
    A1 = a * a * a * a1_shared *a1_shared
    A2 = a * a* a2_shared
    return A1, A2

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

    # topology_file_star = '../Results_Saving/Small_Scale_Network/topo_star'
    # topology_file_P2P = '../Results_Saving/Small_Scale_Network/topo_P2P'
    topology_file_tree = '../Results_Saving/Small_Scale_Network/topo_tree'
    topology_file_mesh = '../Results_Saving/Small_Scale_Network/topo_mesh'

    # coordinates_file 暂时没用到
    coordinates_file = 'Node_Coordinates_100_Uniform[for_priority_analysis]'

    # app_file_star = '../Results_Saving/Small_Scale_Network/App_3_star'
    # app_file_P2P = '../Results_Saving/Small_Scale_Network/App_3_P2P'
    app_file_tree = '../Results_Saving/Small_Scale_Network/App_2_tree'
    app_file_mesh = '../Results_Saving/Small_Scale_Network/App_2_mesh'

    # G_star, Apps_star = init_function_from_file(topology_file_star, coordinates_file, app_file_star, Network_parameters,
    #                                           Wireless_parameters, Loss_parameters)
    # G_P2P, Apps_P2P = init_function_from_file(topology_file_P2P, coordinates_file, app_file_P2P, Network_parameters,
    #                                           Wireless_parameters, Loss_parameters)
    G_tree, Apps_tree = init_function_from_file(topology_file_tree, coordinates_file, app_file_tree, Network_parameters,
                                              Wireless_parameters, Loss_parameters)
    G_mesh, Apps_mesh = init_function_from_file(topology_file_mesh, coordinates_file, app_file_mesh, Network_parameters,
                                              Wireless_parameters, Loss_parameters)

    # 业务可用性评估的参数
    T = 8760*10
    MTTF, MLife = 2000, 800
    MTTR = 4
    ## 重路由相关的参数
    message_processing_time = 0 #0.05  # 单位为秒s [毫秒量级]
    path_calculating_time = 0 #.5  # 单位为秒 s [秒量级]
    detection_rate = 1
    demand_th = 2*math.pow((1/1),1)*math.exp(-1)  # 根据App_demand中的均值来确定,表示性能损失占业务带宽需求的比例
    beta_list = [1]  # 2类可用性指标的权重(beta越大表明 时间相关的服务可用性水平越重要)
    App_priority_list = [1, 2, 3, 4, 5]
    # app_priority = App_priority_list * int(len(Apps) / len(App_priority_list))
    # random.shuffle(app_priority)
    # for i in range(len(Apps)):  # 将业务的优先级设置为 [1~5]
    #     Apps[i].SLA = app_priority[i]
    # nx.draw_networkx(G_tree, pos=nx.spring_layout(G_tree))
    # plt.show()
    # 收敛性分析的参数
    ave_link_fail = 0
    for e in G_tree.edges:
        link_fail = G_tree.adj[e[0]][e[1]]['fail_rate']
        ave_link_fail += link_fail
        print('链路{}的故障率为{}'.format(e, link_fail))
    ave_link_fail = ave_link_fail/len(G_tree.edges)

    print('整网链路的平均故障率为{}'.format(ave_link_fail))

    N = 200
    for i in range(len(Apps_tree)):  # 将业务的优先级设置为 [1~5]
        # Apps[i].SLA = app_priority[i]
        Apps_tree[i].str = 'Global'
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
    availability_app_1 = temp1[0, :]
    availability_app_2 = temp1[1, :]

    temp2 = whole_results_mesh.to_numpy()
    availability_P2P_whole = temp2[0, :]

    # plt.figure(figsize=(10, 5))
    # plt.subplots(121)
    # stats.probplot(availability_mesh_app_1, dist='norm', plot=plt)
    # # plt.title('Normal service availability Q-Q plot of $s_{1}$')
    # plt.subplots(122)
    # stats.probplot(availability_mesh_app_1, dist='norm', plot=plt)
    # plt.title('Normal service availability Q-Q plot of $s_{2}$')
    # plt.show()

    measured_data = (availability_app_1, availability_app_2)
    draw_QQ_plot(measured_data)

    print('simulated A1 is {}'.format(np.mean(availability_app_1)))
    print('simulated A2 is {} \n'.format(np.mean(availability_app_2)))

    # A1_anly, A2_anly = Mesh_RBD_(MTTF, MTTR, 0.1)
    A1_anly, A2_anly = Tree_RBD(MTTF, MTTR, 0.1)
    print('analytical results is {}, absolute_error of A1 is {}'.format(A1_anly, (A1_anly-np.mean(availability_app_1))/ A1_anly ) )
    print('analytical results is {}, absolute_error of A2 is {}'.format(A2_anly,  (A2_anly-np.mean(availability_app_2))/ A2_anly) )

    # # bins = np.linspace(0.9, 1, 21)
    #
    # fig, ax = plt.subplots()
    # n1, bins1, patches = ax.hist(availability_mesh_app_1, bins=20)
    # n2, bins2, patches2 = ax.hist(availability_P2P_whole, bins=20)
    #
    # ax.plot(bins1[:20], n1, marker='o', color='red', linestyle='--')
    # ax.plot(bins2[:20], n2, marker='o', color='green', linestyle='--')
    #
    #
    # # n, bins, patches = plt.hist(availability_P2P_app_1, bins=30, density=True, stacked=True)
    # plt.show(block=True)
