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

def draw_MTBF_plot(x_data, y_data, filename):
    # 绘制带标签的折线图
    time2 = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M') # 记录数据存储的时间
    fig, ax = plt.subplots()
    fig.subplots_adjust(hspace=0.5) # make a little extra space between the subplots
    colors = ['gold','blue','green','orangered','hotpink']
    for i in range(len(y_data)):
        ax.plot(x_data, y_data[i+1],c=colors[i], label='${}$'.format(i+1)) # i+1表示业务等级

    # ax.set_xlim(x_data[0]-3, x_data[-1]+3)
    ax.set_xlabel('Simulation times of RNEM')
    ax.set_ylabel('Coefficient of Variation')
    plt.legend(loc="upper right", title="Priority")
    plt.savefig(r'.\Pictures_saved\line_plot_{}time={}.jpg'.format(filename, time2), dpi=1200)
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


if __name__ == '__main__':
	pass
