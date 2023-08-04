#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   ：Service_availability -> Draw_data_analysis
@IDE    ：PyCharm
@Author ：Yi Zhiwei
@Date   ：2023/6/19 12:16
@Desc   ：数据结果绘图的代码
=================================================='''

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


def draw_priority_plot(x_data, y_data, analysis_param, filename):
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
    ax.set_xlabel('${}$ of network nodes'.format(analysis_param), fontdict=font)
    ax.set_ylabel('Service availability', fontdict=font)

    y_Locator = MultipleLocator(0.0002) # 设置y轴刻度标签为 0.0001 的倍数
    y_Formatter = FormatStrFormatter('%1.4f') #设置y轴标签文本的格式
    ax.yaxis.set_major_locator(y_Locator)
    ax.yaxis.set_major_formatter(y_Formatter)
    plt.ylim(bottom=0.998, top=1)

    plt.legend(loc="lower right", title="Priority")
    plt.subplots_adjust(left=0.15)

    # plt.savefig(r'.\Pictures_saved\{}_plot_{}time={}.jpg'.format(analysis_param, filename, time2), dpi=1200)
    plt.show()

def draw_performance_plot(x_data, y_data, analysis_param, fileneme):
    # 绘制性能权重分析的折线图
    font = {'family': 'serif',
            'color': 'black',
            'weight': 'normal',
            'size': 14, }  # 字体设置
    Beta_list = [0.1, 0.3, 0.5, 0.7, 0.9]

    # 绘制带标签的折线图
    time = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M')  # 记录数据存储的时间
    fig, ax = plt.subplots()
    fig.subplots_adjust(hspace=0.5)  # make a little extra space between the subplots
    colors = ['gold', 'blue', 'green', 'orangered', 'hotpink']
    for i in range(len(y_data)):
        ax.plot(x_data, y_data[i], c=colors[i], alpha=0.5, marker='o', label='$beta$={}'.format(Beta_list[i]))

    ax.set_xlabel('${}$ of network nodes'.format(analysis_param), fontdict=font)
    ax.set_ylabel('Service availability', fontdict=font)

    y_Locator = MultipleLocator(0.0002) # 设置y轴刻度标签为 0.0001 的倍数
    y_Formatter = FormatStrFormatter('%1.4f') #设置y轴标签文本的格式
    ax.yaxis.set_major_locator(y_Locator)
    ax.yaxis.set_major_formatter(y_Formatter)
    plt.ylim(top=1)

    plt.legend(loc="upper right", title="Priority")
    plt.subplots_adjust(left=0.15)

    # plt.savefig(r'.\Pictures_saved\{}_plot_{}time={}.jpg'.format(analysis_param, filename, time2), dpi=1200)
    plt.show()

def draw_resource_plot(x_data, y_data, analysis_param, filename):
    # 绘制资源需求分析的折线图
    pass



def draw_scatter_plot(x_data, y_data, analysisi_param, filename):
    # 绘制结果的散点图

    plt.scatter(x_data, y_data, alpha=0.8)
    plt.show()

def draw_box(data, filename):
    # 绘制箱线图
    df = pd.DataFrame(data)
    # Usual boxplot
    ax = sns.boxplot(data=df, palette='Set2',linewidth=1.5)
    # 通过stripplot添加分布散点图，jitter设置数据间距
    sns.stripplot(data=df, color="orange", jitter=0.2, size=2.5)
    ax.set_ylabel('Availability')
    ax.set_xlabel('Priority of applications')
    time2 = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M') # 记录数据存储的时间
    plt.savefig(r'.\Pictures_saved\boxplot_{}time={}.jpg'.format(filename, time2), dpi=1200)
    plt.grid()
    plt.show()

def data_fitting(x_data_original, y_data_original, file_name):
    # 数据拟合
    xdata = np.array(x_data_original)
    ydata = np.array(y_data_original)

    # 定义使用的公式|customize equation
    def lnFunction(x, A, B):
        return A * np.log(x) + B

    guess = [1, 1]  # 定义初始A、B|initialize a and b
    try:
        params, params_covariance = optimization.curve_fit(lnFunction, xdata, ydata,  guess)  # 拟合，A、B结果存入params|curve fitting and store a, b values to params
        print(params)
        result = ''  # 输出结果|to store result
        for i in range(1, 15):
            result += str(round(lnFunction(i, params[0], params[1]), 2))  # 将i带入公式中的x，使用拟合出的A、B值计算y值，并保留两位小数|calculate result for each i as x using the a, b values, and round the result to 2 points
            if i != 14:
                result += ','  # 每个结果用逗号隔开，并省略最后一个逗号|separate each result with comma, and omit the last comma
        print(result)
    except:
        print('')

if __name__ == '__main__':
    # 对MTTF参数的敏感性分析结果进行绘图
    folder_name = 'Beijing713'
    file_name = 'MTTF敏感性分析-不同优先级的服务可用度-Local策略,演化N=50次,100节点的拓扑_2023_07_10_+03_19'

    x_data, y_data = read_data_from_excel(folder_name, file_name)
    draw_priority_plot(x_data, y_data,'MTTF', 'MTTF敏感性分析, Local策略')

    # data_fitting(x_data, y_data, 'MTTF优先级敏感性分析,Local策略')
    #
    # file_name_global = 'MTTF敏感性分析-不同优先级的服务可用度-Global策略,演化N=50次,100节点的拓扑_2023_07_11_+08_50'
    #
    # x_data_global, y_data_global = read_data_from_excel(folder_name, file_name_global)
    # draw_priority_plot(x_data_global, y_data_global,'MTTF', 'MTTF敏感性分析, Global策略')

    file_name_rate = 'RecoveryRate敏感性分析-不同优先级的服务可用度-Global策略,演化N=50次,100节点的拓扑_2023_07_13+07_45'
    x_data_mttr, y_data_mttr = read_data_from_excel(folder_name, file_name_rate)
    draw_priority_plot(x_data_mttr, y_data_mttr,'recovery_rate', 'MTTR敏感性分析, Global策略')
