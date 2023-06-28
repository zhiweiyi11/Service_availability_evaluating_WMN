#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   ：Service_availability -> Application_request_generating
@IDE    ：PyCharm
@Author ：Yi Zhiwei
@Date   ：2023/3/9 16:18
@Desc   ： 根据给定区域的面积，生成用户业务请求的空间分布
=================================================='''
import copy

import networkx as nx
import numpy as np
import math
import matplotlib.pyplot as plt
import random

import pandas as pd

from Evolution_Model.Evolution_Objects import *


def generate_app_traffic(NB_apps, Area_width, Area_length):
    # 根据区域范围，生成节点的坐标
    pos_x = np.random.uniform(0, Area_width, NB_apps)
    pos_y = np.random.uniform(0, Area_length, NB_apps)
    # 将横纵坐标进行合成为array, 生成节点坐标的阵列
    sensor_coordinates = np.column_stack((pos_x, pos_y))

    return sensor_coordinates

def getDistance(x1, y1, x2, y2): # 计算2点之间的欧氏距离
    distance = 0

    x = x1 - x2
    y = y1 - y2
    distance = math.sqrt(x * x + y * y)

    return distance


"""
@function: 生成网格（或单点）的流量密度，该流量密度服从log-normally分布
@parameters: maxW         : maximum spatial spread
             x and y      : 网格（或单点）的坐标
             sigma and mu : 用于调整统计特征的参数
@return: 单个网格的流量密度
"""
def generateTrafficDensity(maxW, x, y, sigma, mu):
    L = 10  # "L = 10 is large enough" form the ref paper

    # angular frequencies i and j are uniform random variables in [0, maxW]
    i = np.random.uniform(0, maxW, L)
    j = np.random.uniform(0, maxW, L)

    # phases alpha and beta are uniform random variables in [0, 2 * pi]
    alpha = np.random.uniform(0, 2 * math.pi, L)
    beta = np.random.uniform(0, 2 * math.pi, L)

    # Gaussian random field
    sum = 0
    for k in range(L):
        sum += math.cos(i[k] * x + alpha[k]) * math.cos(j[k] * y + beta[k])

    rho = 2 * sum / math.sqrt(L)

    # traffic density ~ log-normally distribution
    density = math.exp(sigma * rho + mu)

    return rho


def generateAppTraffic(Area_width, Area_length, grid_size, traffic_th):
    # 业务请求的空间分布，低于流量请求阈值的业务视为背景流量，仅将超过阈值的流量视为端到端的连通流量
    # 参数初始化
    maxW, sigma, mu = 0.001202, 11.5, 2.3
    AppTraficDensity, GridTrafficDensity = {}, {} # 存储满足业务阈值以及网格对应的业务流量
    for i in range(int(Area_width/grid_size)):
        for j in range(int(Area_length/grid_size)):
            x = grid_size * (i + 0.5) # 当前网格中心的x坐标
            y = grid_size * (j + 0.5) # 当前网格中心的y坐标
            appTraffic = generateTrafficDensity(maxW, x, y, sigma, mu)
            GridTrafficDensity[(x,y)] = appTraffic
            if appTraffic >= traffic_th:
                AppTraficDensity[(x,y)] = appTraffic

    return AppTraficDensity, GridTrafficDensity

def generateAppOD_from_grid(AppTrafficDensity, App_num):
    '''
    :function: 从给定的业务数量中生成业务的源宿节点对
    :param AppTrafficDensity:
    :param App_num:
    :return: 业务的网格坐标以及源宿节点对
    '''
    appTrafic = list(AppTrafficDensity)
    App_coordinates = {}
    App_demand = {}
    for i in range(App_num):
        # 从appTrafficDensity中不重复抽取2个节点作为业务的od
        app_od = random.sample(appTrafic, 2)
        app_demand = 1/2 * (AppTrafficDensity[app_od[0]] + AppTrafficDensity[app_od[1]])
        App_coordinates[i] = app_od
        App_demand[i] = app_demand
        for od in app_od:
            appTrafic.remove(od) # 移除掉已经选择的od对

    return App_coordinates, App_demand


def access_mapping(Node_coordinates, App_coordinates, CV_Range):
    # 根据节点的坐标和业务网格的中心点坐标之间的距离判断来得到业务的映射的拓扑节点集合
    sourceAccessList, terminationAccessList = {}, {}
    for ap_key, ap_value in App_coordinates.items():
        souNodeList, terNodeList = [], [] # 分别存储业务源宿节点的接入网络节点的列表
        for node_key, node_value in Node_coordinates.items():
            node_x, node_y = node_value[0], node_value[1]
            ap_sou_x, ap_sou_y = ap_value[0][0], ap_value[0][1] # 业务的坐标dict中存储了其源宿节点grid 的坐标
            ap_ter_x, ap_ter_y = ap_value[1][0], ap_value[1][1] # 所以要分别读取和计算

            distance_sou = getDistance(node_x, node_y, ap_sou_x, ap_sou_y)
            distance_ter = getDistance(node_x, node_y, ap_ter_x, ap_ter_y)
            if distance_sou <= CV_Range:
                souNodeList.append(node_key)
            if distance_ter <= CV_Range:
                terNodeList.append(node_key)

        # 去除业务接入和接出集合中重复的节点
        remove_list = []
        for node in terNodeList:
            if node in souNodeList:
                # print('当前业务的接入节点集合{}和退出节点集合{}'.format(souNodeList, terNodeList))
                remove_list.append(node)
        # 根据接入和接出节点list的长度，来对其进行去重操作
        for n in remove_list:
            if len(souNodeList) > len(terNodeList):
                # 如果接入节点集的长度大于接出节点集
                souNodeList.remove(n)
            else:
                terNodeList.remove(n)

        sourceAccessList[ap_key] = souNodeList
        terminationAccessList[ap_key] = terNodeList


    return sourceAccessList, terminationAccessList




if __name__ == '__main__':
    # 完成网络的代码调试
    Node_num =  100
    App_num = 100
    Cap_node = 20
    Topology = 'Random'
    Area_width, Area_length = 250, 250
    TX_range = 50 # 传输范围为区域面积的1/5时能够保证网络全联通
    CV_range = 30 # 节点的覆盖范围
    grid_size = 5
    traffic_th = 1

    # Coordinates = generate_positions(Node_num, Area_width, Area_length)
    #
    # G = Network(Topology, Node_num, Coordinates, TX_range, True)
    # Node_Coordinates = {} # 存储节点的坐标信息
    # for n in list(G):
    #     Node_Coordinates[n] = Coordinates[n]
    # G.set_node_attributes( Node_Coordinates) # 每个节点的容量相同

    TrafficDensity = generateAppTraffic(Area_width, Area_length, grid_size, traffic_th) # 生成业务流量分布,从网格中找到业务的请求
    App_coordinates, App_demand = generateAppOD_from_grid(TrafficDensity[0], App_num) # 根据各网格上的流量密度和业务请求的数量,找到业务请求对应的od和demand
    df = pd.DataFrame(columns=['coordinates', 'demand'])
    # 将业务流量请求的数据保存下来
    for i in range(len(App_demand)):
        coord = App_coordinates[i]
        demand = App_demand[i]
        df.loc[i] = [coord, demand]
    saveDataFrameToExcel(df, 'AppTraffic_100_SINR')




    # App_access = access_mapping(Node_Coordinates, App_coordinates, CV_range)
    # Access = App_access[0]
    # Exit = App_access[1]
    # for i in range(len(Access)):
    #     for node in Access[i]:
    #         if node in Exit[i]:
    #             print('业务{}的接入{}和接出{}存在重复节点{}'.format(i, Access[i], Exit[i], node))
    #             break

