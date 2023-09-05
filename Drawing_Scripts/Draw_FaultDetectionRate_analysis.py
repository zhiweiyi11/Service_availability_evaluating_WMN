#!/usr/bin/env python
# -*- coding: utf-8 -*-
''' =====================================
# @Time    : 2023/9/4 21:17
# @Author  : Yi Zhiwei
# @IDE     : PyCharm
# @File    : Draw_FaultDetectionRate_analysis.py
# @Software: PyCharm
# Desc     : 绘制“故障检测率”对服务可用度的敏感性分析
======================================'''


import networkx as nx
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import matplotlib as mpl
from matplotlib.collections import PatchCollection
from matplotlib.pyplot import MultipleLocator, FormatStrFormatter
from scipy import stats
import scipy.optimize as optimization

def read_data_from_excel(folder_name, file_name):
    # 根据文件夹的名称和文件的名称，读取服务可用性的结果
    availability_data = pd.read_excel(r"..\Drawing_Scripts\Data_saved\{}\{}.xlsx".format(folder_name, file_name))
    x_data = availability_data.columns.to_list()
    # y_data = availability_data.to_dict() # 将每一列的数据存储为一个字典
    y_data = {}
    for index in availability_data.index.to_list():
        y_data[index] = availability_data.loc[index,:].to_list()

    return x_data, y_data
def draw_priority_analysis(x_data, y_data, analysis_param, filename):
    font = {'family': 'serif',
            'color': 'black',
            'weight': 'normal',
            'size': 14,  } # 字体设置

    # 绘制带标签的折线图
    time2 = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M') # 记录数据存储的时间
    fig, ax = plt.subplots()
    fig.subplots_adjust(hspace=0.5) # make a little extra space between the subplots
    colors = ['gold','blue','green','orangered','hotpink']
    for i in range(len(y_data)):
        ax.plot(x_data, y_data[i],c=colors[i],alpha = 0.5,marker='o', label='${}$'.format(i+1)) # i+1表示业务等级

    # ax.set_xlim(x_data[0]-3, x_data[-1]+3)
    ax.set_xlabel(r'Fault detection rate $\rho$ ', fontdict=font)
    ax.set_ylabel('Service availability', fontdict=font)

    y_Locator = MultipleLocator(0.0005) # 设置y轴刻度标签为 0.0001 的倍数
    y_Formatter = FormatStrFormatter('%1.4f') #设置y轴标签文本的格式
    ax.yaxis.set_major_locator(y_Locator)
    ax.yaxis.set_major_formatter(y_Formatter)
    # plt.xlim(left=0.8, right=0.99)
    plt.ylim(bottom=0.995, top=1) # Local策略下设置为0.9992，使得不同priority之间的结果更明显

    plt.legend(loc="lower right",title="Priority") # loc="lower right",
    plt.subplots_adjust(left=0.15)

    plt.savefig(r'..\Pictures_saved\Restoration_Rate\{}_plot_{}time={}.jpg'.format(analysis_param, filename, time2), dpi=1200)
    plt.show()

def draw_resource_plot(x_data, y_data, analysis_param, filename):
    # 绘制资源需求分析的折线图
    font = {'family': 'serif',
            'color': 'black',
            'weight': 'normal',
            'size': 14, }  # 字体设置
    marker_list = ['o', 'v', '^', '*']
    # 绘制带标签的折线图
    time = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M')  # 记录数据存储的时间
    fig, ax = plt.subplots()
    fig.subplots_adjust(hspace=0.5)  # make a little extra space between the subplots
    colors = ['goldenrod', 'darkred', 'royalblue','purple' ]
    for i in range(len(y_data)):
        ax.plot(x_data[2:], y_data[i][2:], c=colors[i], alpha=0.5, marker=marker_list[i], label='$\mathcal{B}$='+'{},'.format(int(y_data[i][0]))+'$\mathcal{D}$='+'{}'.format(int(y_data[i][1]))) #$\mathcal{B}$=$\mathcal{D}$

    ax.set_xlabel(r'Fault detection rate $\rho$'.format('MTTF'), fontdict=font)
    ax.set_ylabel('Service availability', fontdict=font)

    y_Locator = MultipleLocator(0.001)  # 设置y轴刻度标签为 0.0001 的倍数
    y_Formatter = FormatStrFormatter('%1.4f')  # 设置y轴标签文本的格式
    ax.yaxis.set_major_locator(y_Locator)
    ax.yaxis.set_major_formatter(y_Formatter)
    plt.ylim(bottom=0.993,top=1)

    plt.legend(loc='lower right') # loc="upper right",
    plt.subplots_adjust(left=0.15)

    plt.savefig(r'..\Pictures_saved\Restoration_Rate\{}_plot_{}time={}.jpg'.format(analysis_param, filename, time), dpi=1200)
    plt.show()


if __name__ == '__main__':
    folder_name = 'FaultDetectionRate'
    file_name = 'RecoveryRate敏感性分析-不同优先级的服务可用度-Global策略,演化N=50次,100节点的拓扑_2023_08_15+08_13'

    x_data_Global, y_data_Global = read_data_from_excel(folder_name, file_name)
    # draw_priority_analysis(x_data_Global, y_data_Global, 'FaultDetectionRate优先级分析', 'Global策略')

    file_name_2 = 'RecoveryRate敏感性分析-不同优先级的服务可用度-Local策略,演化N=50次,100节点的拓扑_2023_08_14+22_11'

    x_data_Local, y_data_Local = read_data_from_excel(folder_name, file_name_2)
    # draw_priority_analysis(x_data_Local, y_data_Local, 'FaultDetectionRate优先级分析', 'Local策略')

    file_resource_global = 'RecoveryRate敏感性分析-不同网络规模汇总-Global策略-20230814'
    file_resource_local = 'RecoveryRate敏感性分析-不同网络规模汇总-Local策略-20230814'


    x_local, y_local = read_data_from_excel( folder_name, file_resource_local)
    # draw_resource_plot(x_local, y_local,'RecoveryRate资源可用性分析', 'Local策略')

    x_global, y_global = read_data_from_excel(folder_name, file_resource_global)
    draw_resource_plot(x_global, y_global, 'RecoveryRate资源可用性分析', 'Global策略')

