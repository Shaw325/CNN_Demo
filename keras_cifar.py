# -*- coding: utf-8 -*-
"""
-------------------------------------------------
         File Name      : keras_cifar
         Description    : todo
         Author         : lindsey
         date           : 2021/7/18 21:26
-------------------------------------------------
         Change Activity:
             2021/7/18 21:26: todo
-------------------------------------------------
"""

"""

+ airplane              0
+ automobile            1
+ bird                  2
+ cat                   3
+ deer                  4
+ dog                   5
+ frog                  6
+ horse                 7
+ ship                  8
+ trunk                 9

"""
import tensorflow.keras as keras
import tensorflow.keras.datasets.cifar10 as cifar
from matplotlib import pyplot as plt
import numpy as np

label_dict = {
    0: "飞机",
    1: "汽车",
    2: "鸟",
    3: "猫",
    4: "鹿",
    5: "狗",
    6: "青蛙",
    7: "马",
    8: "船",
    9: "卡车"
}


# 加载数据集，分为测试数据集和训练数据集
def load_date():
    return cifar.load_data()


# 打印图片
def show(image):
    plt.imshow(image)
    plt.show()


(x_train, y_train), (x_test, y_test) = load_date()

# show(x_train[0])

print("Train data set, Total:", len(x_train))
print("Test data set, Total:", len(x_test))

# 整理数据集,
y_train = np.hstack(y_train)
y_test = np.hstack(y_test)


