#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   ：Service_availability -> Draw_boxplot
@IDE    ：PyCharm
@Author ：Yi Zhiwei
@Date   ：2023/7/17 18:08
@Desc   ：根据节点的密度分布来绘制箱线图
=================================================='''

import seaborn as sns
sns.set_theme(style="ticks", palette="pastel")

# Load the example tips dataset
tips = sns.load_dataset("tips")

# Draw a nested boxplot to show bills by day and time
sns.boxplot(x="day", y="total_bill",
            hue="smoker", palette=["m", "g"],
            data=tips)
sns.despine(offset=10, trim=True)



if __name__ == '__main__':
	pass
