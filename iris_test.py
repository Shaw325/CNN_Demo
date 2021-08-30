# -*- coding: utf-8 -*-
"""
-------------------------------------------------
         File Name      : iris_test
         Description    : todo
         Author         : lindsey
         date           : 2021/6/28 1:07
-------------------------------------------------
         Change Activity:
             2021/6/28 1:07: todo
-------------------------------------------------
"""
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import numpy as np
from pandas import DataFrame
import pandas as pd
import tensorflow as tf


def load_data():
    # 加载150条鸢尾花数据
    x_data = load_iris().data
    y_data = load_iris().target
    return x_data, y_data


def slice_train_test_data(x_data, y_data):
    train_x = x_data[:-30]
    test_x = x_data[-30:]
    train_y = y_data[:-30]
    test_y = y_data[-30:]
    return train_x, train_y, test_x, test_y


x_data, y_data = load_data()

# 随机种子，使得每个鸢尾花特征向量可以和一个y对应
np.random.seed(10)
np.random.shuffle(x_data)
np.random.seed(10)
np.random.shuffle(y_data)

print("乱序后")
# 转换为DataView模式
data_set = DataFrame(x_data, columns=['花萼长度', '花萼宽度', '花瓣长度', '花瓣宽度'])
pd.set_option('display.unicode.east_asian_width', True)  # 列名对齐
# 增加一列-【类别】
data_set['类别'] = y_data

# 将数据集划分为训练集和测试集，为防止过拟合，训练集与测试集比例最好为80%/20%
train_x, train_y, test_x, test_y = slice_train_test_data(x_data, y_data)

BATCH_SIZE = 30

# 将数据集按批次打包
train_x = tf.cast(train_x, tf.float32)
test_x = tf.cast(test_x, tf.float32)

train_db = tf.data.Dataset.from_tensor_slices((train_x, train_y)).batch(BATCH_SIZE)
test_db = tf.data.Dataset.from_tensor_slices((test_x, test_y)).batch(BATCH_SIZE)

"""
一、建立神经网络
    鸢尾花具备四个特征，三种类型
    建立的神经网络应该是输入层为4，输出层为3的结构
二、设置回归函数
    y = x*w + b    
三、梯度下降-参数修正
   w = w - lr * 【w/loss】复合求导
   b = b - lr * 【b/loss】复合求导
"""
# 学习率
lr = 0.1
# 迭代次数-训练500次
epoch = 500
# 记录loss变化曲线
train_loss_set = []
# 测试准确度
test_acc = []
loss_all = 0

w = tf.Variable(tf.random.truncated_normal([4, 3], stddev=0.1, seed=1))
b = tf.Variable(tf.random.truncated_normal([3], stddev=0.1, seed=1))

"""
    嵌套循环迭代，with结构语法
    梯度下降法
"""
for epoch in range(epoch):
    for step, (train_x, train_y) in enumerate(train_db):
        with tf.GradientTape() as tape:
            # 求出y的值
            y = tf.matmul(train_x, w) + b
            # softmax函数 将y值映射到【0，1】区间，即归一化
            y = tf.nn.softmax(y)
            y_ = tf.one_hot(train_y, depth=3)
            # 实际y值与前向传播y值的均方误差，最小二乘法
            loss = tf.reduce_mean(tf.square(y_ - y))
            # 累加所有的损失值
            loss_all += loss.numpy()
        # 根据损失值计算参数梯度,对w和b分别求导
        grads = tape.gradient(loss, [w, b])

        # 梯度下降对参数进行修正,拟合至loss最小值
        w.assign_sub(lr * grads[0])
        b.assign_sub(lr * grads[1])
    print("Epoch {}, loss: {}".format(epoch, loss_all/4))
    train_loss_set.append(loss_all / 4)
    loss_all = 0

    total_correct, total_number = 0, 0
    for test_x, test_y in test_db:
        y = tf.matmul(test_x, w) + b
        # 归一化
        y = tf.nn.softmax(y)
        # 最接近的分类
        pred = tf.argmax(y, axis=1)
        pred = tf.cast(pred, dtype=test_y.dtype)
        correct = tf.cast(tf.equal(pred, test_y), dtype=tf.int32)
        correct = tf.reduce_sum(correct)
        total_correct += int(correct)
        total_number += test_x.shape[0]
    acc = total_correct / total_number
    test_acc.append(acc)
    print("Test_acc:", acc)
    print("===========================")


"""
总结：
    使用梯度下降优化参数
    参数通过 w = w - lr * [loss]求导 
    lr率越低，损失函数曲线越平滑
    lr越低，学习精准率曲线越平滑
    
"""
plt.title("Loss Function Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.plot(train_loss_set, label="$Loss$")
plt.legend()
plt.show()

plt.title("Acc Function Curve")
plt.xlabel("Epoch")
plt.ylabel("Acc")
plt.plot(test_acc, label="$acc$")
plt.legend()
plt.show()

