
# coding: utf-8

#数据挖掘作业1——李盛楠——2120161009
#——————————————————————————————————————————————————————————————————————————————————————————————————————————————————
#相关作业要求如下：
#——————————————————————————————————————————————————————————————————————————————————————————————————————————————————
#1. 问题描述
#疝病是描述马胃肠痛的术语，这种病不一定源自马的胃肠问题，其他问题也可能引发马疝病。所给数据集是医院检测的一些指标。
#——————————————————————————————————————————————————————————————————————————————————————————————————————————————————
#2. 数据说明
#下载数据: 地址
#共368个样本，27个特征。关于特征的详细说明见下载链接。
#——————————————————————————————————————————————————————————————————————————————————————————————————————————————————
#3. 数据分析要求
#3.1 数据可视化和摘要
#数据摘要
#•	对标称属性，给出每个可能取值的频数，
#•	数值属性，给出最大、最小、均值、中位数、四分位数及缺失值的个数。
#数据的可视化
#针对数值属性，
#•	绘制直方图，如mxPH，用qq图检验其分布是否为正态分布。
#•	绘制盒图，对离群值进行识别
#3.2 数据缺失的处理
#数据集中有30%的值是缺失的，因此需要先处理数据中的缺失值。
#分别使用下列四种策略对缺失值进行处理:
#•	将缺失部分剔除
#•	用最高频率值来填补缺失值
#•	通过属性的相关关系来填补缺失值
#•	通过数据对象之间的相似性来填补缺失值
#处理后，可视化地对比新旧数据集。
#——————————————————————————————————————————————————————————————————————————————————————————————————————————————————
#4. 提交内容
#•	分析过程的报告
#•	分析程序
#•	预处理后的数据集
#——————————————————————————————————————————————————————————————————————————————————————————————————————————————————


import operator
import numpy as np
import scipy.stats as stats

#使用pandas库进行数据分析
import pandas as pd
import statsmodels.api as sm

#使用matplotlib库进行数据的可视化处理
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')


#数据文件格式转换，从txt转化为csv格式
fp_origin = open("./data.txt", 'r')
fp_modified = open("./data.csv", 'w')

line = fp_origin.readline()
while(line):
    temp = line.strip().split()
    temp = ','.join(temp)+'\n'
    fp_modified.write(temp)
    line = fp_origin.readline()
    
fp_origin.close()
fp_modified.close()

#数据摘要
#•  对标称属性，给出每个可能取值的频数，
#•  数值属性，给出最大、最小、均值、中位数、四分位数及缺失值的个数。
#读取csv文件，完成数据摘要工作

# 根据数据集介绍，定义数据属性的名称
attribute=["surgery"," Age ","Hospital Number","rectal temperature","pulse ","respiratory rate "," temperature of extremities","peripheral pulse","mucous membranes","capillary refill time "," pain","peristalsis "," abdominal distension","nasogastric tube ","nasogastric reflux "," nasogastric reflux PH "," rectal examination"," abdomen "," packed cell volume "," total protein "," abdominocentesis appearance "," abdomcentesis total protein","outcome ","surgical lesion"," lesion 1"," lesion 2"," lesion 3","cp_data "]

# 读取数据
data_origin = pd.read_csv("./data.csv", 
                   names = attribute,
                   na_values = "?")

# 对标称属性进行频数的统计，写入frequency.txt文件
# 使用value_counts函数统计每个标称属性的取值频数

for item in attribute:
    output = open('D:\LSN\DM\\frequency.txt','a')
    output.write(str(item))
    output.write('\n')
    output.write(str(pd.value_counts(data_origin[item].values)))
    output.write('\n')
    output.write('\n')
output.close()

# 对数值属性，统计最大值、最小值、均值、中位数、四分位数及缺失值的个数
# 将结果写入numerical_attribute.txt文件

# 最大值
data_show = pd.DataFrame(data = data_origin[attribute].max(), columns = ['max'])

# 最小值
data_show['min'] = data_origin[attribute].min()

# 均值
data_show['mean'] = data_origin[attribute].mean()

# 中位数
data_show['median'] = data_origin[attribute].median()

# 四分位数
data_show['quartile'] = data_origin[attribute].describe().loc['25%']

# 缺失值的个数
data_show['missing'] = data_origin[attribute].describe().loc['count'].apply(lambda x : 200-x)

output = open('D:\LSN\DM\\numerical_attribute.txt','a')
output.write(str(data_show))

#数据的可视化
#针对数值属性，
#•  绘制直方图，如mxPH，用qq图检验其分布是否为正态分布。
#•  绘制盒图，对离群值进行识别

# 直方图
fig = plt.figure(figsize = (30,20))
i = 1
for item in attribute:
    ax = fig.add_subplot(4,7,i)
    data_origin[item].plot(kind = 'hist', title = item, ax = ax)
    i += 1
plt.subplots_adjust(wspace = 0.3, hspace = 0.3)
fig.savefig('./image/histogram.pdf')
print ('histogram saved at ./image/histogram.pdf')


# qq图
fig = plt.figure(figsize = (30,20))
i = 1
for item in attribute:
    ax = fig.add_subplot(4,7,i)
    sm.qqplot(data_origin[item], ax = ax)
    ax.set_title(item)
    i += 1
plt.subplots_adjust(wspace = 0.3, hspace = 0.3)
fig.savefig('./image/qqplot.pdf')
print ('qqplot saved at ./image/qqplot.pdf')


# 盒图
fig = plt.figure(figsize = (30,20))
i = 1
for item in attribute:
    ax = fig.add_subplot(4,7,i)
    data_origin[item].plot(kind = 'box')
    i += 1
fig.savefig('./image/boxplot.pdf')
print ('boxplot saved at ./image/boxplot.pdf')

#3.2 数据缺失的处理
#数据集中有30%的值是缺失的，因此需要先处理数据中的缺失值。
#分别使用下列四种策略对缺失值进行处理:
#•  将缺失部分剔除
#•  用最高频率值来填补缺失值
#•  通过属性的相关关系来填补缺失值
#•  通过数据对象之间的相似性来填补缺失值
#处理后，可视化地对比新旧数据集。

# 找出含有缺失值的数据条目索引值
nan_list = pd.isnull(data_origin).any(1).nonzero()[0]

# 使用策略一：
# 剔除含有缺失值的数据条目，生成新的数据集

# 使用dropna()函数操作删除缺失值
data_filtrated = data_origin.dropna()

# 绘制可视化图
fig = plt.figure(figsize = (30,20))

i = 1

# 绘制直方图
for item in attribute:
    ax = fig.add_subplot(4,7,i)
    ax.set_title(item)
    data_origin[item].plot(ax = ax, alpha = 0.3, kind = 'hist', label = 'origin', legend = True)
    data_filtrated[item].plot(ax = ax, alpha = 0.8, kind = 'hist', label = 'filtrated', legend = True)
    ax.axvline(data_origin[item].mean(), color = 'r')
    ax.axvline(data_filtrated[item].mean(), color = 'b')
    i += 1
plt.subplots_adjust(wspace = 0.3, hspace = 0.3)

# 保存图像和处理后数据
fig.savefig('./image/delete_missing_data.pdf')
data_filtrated.to_csv('./data_output/delete_missing_data.csv', mode = 'w', encoding='utf-8', index = False,header = False)
print ('filted_missing_data1 saved at ./image/delete_missing_data.pdf')
print ('data after analysis saved at ./data_output/delete_missing_data.csv')

# 使用策略二：
# 用最高频率值来填补缺失值，生成新的数据集


# 使用value_counts()函数统计原始数据中，出现频率最高的值
# 再用fillna()函数将缺失值替换为最高频率值。
# 建立原始数据的拷贝
data_filtrated = data_origin.copy()

# 分别对每一列数据进行处理
for item in attribute:
    # 计算出最高频率的值
    most_frequent_value = data_filtrated[item].value_counts().idxmax()
    # 替换缺失值
    data_filtrated[item].fillna(value = most_frequent_value, inplace = True)

# 绘制可视化图
fig = plt.figure(figsize = (30,20))

i = 1

# 绘制直方图
for item in attribute:
    ax = fig.add_subplot(4,7,i)
    ax.set_title(item)
    data_origin[item].plot(ax = ax, alpha = 0.8, kind = 'hist', label = 'origin', legend = True)
    data_filtrated[item].plot(ax = ax, alpha = 0.3, kind = 'hist', label = 'droped', legend = True)
    ax.axvline(data_origin[item].mean(), color = 'r')
    ax.axvline(data_filtrated[item].mean(), color = 'b')
    i += 1
plt.subplots_adjust(wspace = 0.3, hspace = 0.3)

# 保存图像和处理后数据
fig.savefig('./image/most_frequency_missing_data.pdf')
data_filtrated.to_csv('./data_output/most_frequency_missing_data.csv', mode = 'w', encoding='utf-8', index = False,header = False)
print ('filted_missing_data2 saved at ./image/most_frequency_missing_data.pdf')
print ('data after analysis saved at ./data_output/most_frequency_missing_data.csv')

# 使用策略三：
# 通过属性的相关关系来填补缺失值，生成新的数据集

# 使用pandas中Series的***interpolate()***函数，对数值属性进行插值计算，并替换缺失值。
# 建立原始数据的拷贝
data_filtrated = data_origin.copy()

#进行插值运算
for item in attribute:
    data_filtrated[item].interpolate(inplace = True)

# 绘制可视化图
fig = plt.figure(figsize = (30,20))

i = 1

# 绘制直方图
for item in attribute:
    ax = fig.add_subplot(4,7,i)
    ax.set_title(item)
    data_origin[item].plot(ax = ax, alpha = 0.5, kind = 'hist', label = 'origin', legend = True)
    data_filtrated[item].plot(ax = ax, alpha = 0.5, kind = 'hist', label = 'droped', legend = True)
    ax.axvline(data_origin[item].mean(), color = 'r')
    ax.axvline(data_filtrated[item].mean(), color = 'b')
    i += 1
plt.subplots_adjust(wspace = 0.3, hspace = 0.3)

# 保存图像和处理后数据
fig.savefig('./image/corelation_missing_data.pdf')
data_filtrated.to_csv('./data_output/corelation_missing_data.csv', mode = 'w', encoding='utf-8', index = False,header = False)
print ('filted_missing_data3 saved at ./image/corelation_missing_data.pdf')
print ('data after analysis saved at ./data_output/corelation_missing_data.csv')

# 使用策略四：
# 通过数据对象之间的相似性来填补缺失值，生成新的数据集

# 先将缺失值设为0，对数据集进行正则化。
# 再对每两条数据进行差异性计算（分值越高差异性越大）。
# 计算标准为：标称数据不相同记为1分，数值数据差异性分数为数据之间的差值。
# 在处理缺失值时，找到和该条数据对象差异性最小（分数最低）的对象，将最相似的数据条目中对应属性的值替换缺失值。

# 建立原始数据的拷贝，用于正则化处理
data_norm = data_origin.copy()

# 将数值属性的缺失值替换为0
data_norm[attribute] = data_norm[attribute].fillna(0)

# 对数据进行正则化
data_norm[attribute] = data_norm[attribute].apply(lambda x : (x - np.mean(x)) / (np.max(x) - np.min(x)))

# 构造分数表
score = {}
range_length = len(data_origin)
for i in range(0, range_length):
    score[i] = {}
    for j in range(0, range_length):
        score[i][j] = 0    

# 对处理后的数据进行两两对比，每两条数据条目计算差异性得分，分值越高差异性越大
for i in range(0, range_length):
    for j in range(i, range_length):
        for item in attribute:
            temp = abs(data_norm.iloc[i][item] - data_norm.iloc[j][item])
            score[i][j] += temp
        score[j][i] = score[i][j]

# 建立原始数据的拷贝
data_filtrated = data_origin.copy()

# 对有缺失值的条目，用和它相似度最高（得分最低）的数据条目中对应属性的值替换
for index in nan_list:
    best_friend = sorted(score[index].items(), key=operator.itemgetter(1), reverse = False)[1][0]
    for item in attribute:
        if pd.isnull(data_filtrated.iloc[index][item]):
            if pd.isnull(data_origin.iloc[best_friend][item]):
                data_filtrated.ix[index, item] = data_origin[item].value_counts().idxmax()
            else:
                data_filtrated.ix[index, item] = data_origin.iloc[best_friend][item]

# 绘制可视化图
fig = plt.figure(figsize = (30,20))

i = 1
# 绘制直方图
for item in attribute:
    ax = fig.add_subplot(4,7,i)
    ax.set_title(item)
    data_origin[item].plot(ax = ax, alpha = 0.5, kind = 'hist', label = 'origin', legend = True)
    data_filtrated[item].plot(ax = ax, alpha = 0.5, kind = 'hist', label = 'droped', legend = True)
    ax.axvline(data_origin[item].mean(), color = 'r')
    ax.axvline(data_filtrated[item].mean(), color = 'b')
    i += 1
plt.subplots_adjust(wspace = 0.3, hspace = 0.3)

# 保存图像和处理后数据
fig.savefig('./image/similarity_missing_data.pdf')
data_filtrated.to_csv('./data_output/similarity_missing_data.csv', mode = 'w', encoding='utf-8', index = False,header = False)
print ('filted_missing_data4 saved at ./image/similarity_missing_data.pdf')
print ('data after analysis saved at ./data_output/similarity_missing_data.csv')

