#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   ：Service_availability -> FunctionTest
@IDE    ：PyCharm
@Author ：Yi Zhiwei
@Date   ：2023/3/12 17:30
@Desc   ：用于测试一些简单的函数功能
=================================================='''
import networkx as nx
import numpy as np
import random


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





if __name__ == '__main__':
    line_List = [['aa', 'bb', 'cc'], ['dd', 'ee', 'ff'], ['gg', 'hh']]
    num_list = [[69, 31, 98], [61, 97, 78] ]

    caseslist = RecursionFunc(line_List[0], line_List[1:])
    numslist = RecursionFunc(num_list[0], num_list[1:])
    for num in numslist:
        print(num)
        print('*******\n')

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