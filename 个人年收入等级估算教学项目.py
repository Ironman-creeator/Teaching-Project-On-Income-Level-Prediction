# 1. 数据读取与初步处理
# 首先使用pandas库读取adult.csv文件，该文件包含 15 列数据（如年龄、单位性质等），并指定了列名。读取后查看了原始数据文件的形态以及输出了前 5 行数据。
# 接着根据 “收入” 列对数据进行分组，分别获取了收入 “<=50K” 和 “>50K” 的分组数据，并统计了这两种收入情况的样本数量。之后选取了收入 “<=50K” 的 10000 个样本和收入 “>50K” 的 7841 个样本构成新的数据集，对其进行合并与排序后查看了数据集形态并输出前 10 行。
# 读取数据
import pandas as pd
# 用pandas打开csv文件
df = pd.read_csv('adult.csv', header=None, index_col=False,
                  names=['年龄','单位性质','权重','学历','受教育时长',
                        '婚姻状况','职业','家庭情况','种族','性别',
                        '资产所得','资产损失','周工作时长','原籍',
                        '收入'])
print('adult文件的数据形态：', df.shape)
print('输出数据的前5行：')

df.groupby(by='收入').agg({'收入': 'count'})


# 根据‘收入’进行分组
group_income = df.groupby(by='收入')
# 收入<=50K的分组
income_lessthan50k = dict([x for x in group_income])[' <=50K']
# 收入>50K的分组
income_morethan50k = dict([x for x in group_income])[' >50K']
print('收入 <=50K 的样本数量：', income_lessthan50k.shape[0])
print('收入 >50K 的样本数量：', income_morethan50k.shape[0])


# - 选取收入<=50K的10000个样本、收入>50K的7841个样本构成数据集
# 合并数据分组并排序
data = pd.concat([income_lessthan50k[:10000], income_morethan50k], axis=0)
data = data.sort_index()
print('数据集形态：', data.shape)
print('输出数据集前10行：')
#
# 2. 数据转换与编码处理
# 介绍了标记编码方法，通过sklearn.preprocessing中的LabelEncoder对示例标记进行编码演示，包括编码、输出编码结果、用编码器转换标记以及逆变换还原标记等操作。
# 针对整个数据集，定义了get_data_encoded函数，将数据集中的字符串数据转换为数值数据（数值数据保持不变），通过遍历数据列判断是否为数值型，若非数值则进行标记编码，最终返回编码后的数据以及编码器列表。利用该函数对前面构建的数据集进行处理，并将处理后的数据集拆分成特征矩阵X和类别标签y。
# - 将数据集中的字符串数据转换为数值数据，同时需要保留原有的数值数据

# #### 标记编码方法
from sklearn.preprocessing import LabelEncoder
# 定义一个标记编码器
label_encoder = LabelEncoder()
# 创建一些标记
input_classes = ['audi', 'ford', 'audi', 'toyota', 'ford', 'bmw']
# 为这些标记编码
label_encoder.fit(input_classes)
# 输出编码结果
print('Class mapping:')
print(label_encoder.classes_)
for i, item in enumerate(label_encoder.classes_):
    print(item, '-->', i)


# 用编码器转换一组标记
labels = ['toyota', 'ford', 'audi']
encoded_labels = label_encoder.transform(labels)
print('Labels = ', labels)
print(encoded_labels)
print('Encoded labels = ', list(encoded_labels))

label_encoder.inverse_transform(encoded_labels)


# #### 对数据集进行标记编码

import numpy as np
from sklearn.preprocessing import LabelEncoder

# 定义一个用于标签编码的函数
def get_data_encoded(data):
    # 将数据全部转为字符类型
    data = np.array(data.astype(str))
    # 定义标记编码器对象列表
    encoder_list = []
    # 准备一个数组存储数据集编码后的结果
    data_encoded = np.empty(data.shape)
    # 将字符串转换为数值数据
    for i, item in enumerate(data[0]):
        # 判断该特征向量是否为数值数据
        if item.isdigit():
            data_encoded[:, i] = data[:, i]
        # 如果不是数值数据则进行标记编码
        else:
            # 将所有的标记编码器保存在列表中，以便在后面测试数据时使用
            encoder_list.append(LabelEncoder())
            # 将字符串数据的特征列逐个进行编码
            data_encoded[:, i] = encoder_list[-1].fit_transform(data[:, i])
    
    # 返回数据编码结果和编码器列表
    return data_encoded, encoder_list

data_encoded, encoder_list = get_data_encoded(data)
# 将编码处理完成的数据集拆分成特征矩阵X和类别标签y
X = data_encoded[:, :-1].astype(int)
# 数据集最后一列“收入”作为分类的类别标签，‘<=50K’为0，‘>50K’为1
y = data_encoded[:, -1].astype(int)
print(encoder_list)
print('编码处理完成的数据集')
print('特征形态：{}，标签形态：{}'.format(X.shape, y.shape))

#
# 3. 高斯朴素贝叶斯建模
# 使用sklearn.naive_bayes中的GaussianNB进行建模。先将数据集拆分为训练集和测试集，然后对数值数据进行预处理（标准化），接着用高斯朴素贝叶斯模型拟合训练集数据，并分别打印出训练集和测试集的得分以评估模型效果。


from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 拆分数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=5)
# 对数值进行预处理
scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 使用高斯朴素贝叶斯拟合数据
gnb = GaussianNB()
gnb.fit(X_train_scaled, y_train)
# 打印模型评分
print('训练集得分：{:.3f}'.format(gnb.score(X_train_scaled, y_train)))
print('测试集得分：{:.3f}'.format(gnb.score(X_test_scaled, y_test)))

#
# 4. 模型预测
# 从原始数据中收入 “<=50K” 且未选入前面数据集的部分选取了 3 个样本作为测试样本，将其与原数据集合并后进行编码处理，获取编码后的测试数据。
# 从中分离出特征数据X和分类y，对特征数据X进行标准化处理，然后使用训练好的模型对测试样本的特征数据进行预测分类标签，最后将预测的分类标签解码还原成原来的数据形式（收入等级）。

# 从数据文件里选择样本做测试
test = income_lessthan50k[10000:10003]
print('选取测试样本：')


# 将测试样本与原数据集进行合并
data_all = pd.concat([data, test])


# 调用标记编码函数进行数据编码
data_all_encoded, encoder_list = get_data_encoded(data_all)
data_all_encoded.astype(int)


# 取出最后3个已经编码的测试数据
test_encoded = data_all_encoded[-3:].astype(int)
print('打印编码转换后的测试数据：\n', test_encoded)


# 获取测试样本的特征数据X和分类y
test_encoded_X = test_encoded[:,:-1]
test_encoded_y = test_encoded[:, -1]
print('测试样本的特征数据X：\n', test_encoded_X)
print('\n测试样本的收入等级：', test_encoded_y)


# 将编码后的特征数据X标准化
test_encoded_X_scaled = scaler.transform(test_encoded_X)
test_encoded_X_scaled


# 对数据进行预测分类标签
pred_encoded_y = gnb.predict(test_encoded_X_scaled)
print('测试样本的预测分类为：', pred_encoded_y)
# 对分类标签进行解码，转换成原来的数据形式
pred_y = encoder_list[-1].inverse_transform(pred_encoded_y)
print('预测的收入等级：', pred_y)




