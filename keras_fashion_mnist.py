# -*- coding: utf-8 -*-
"""
-------------------------------------------------
         File Name      : keras_fashion_mnist
         Description    : todo
         Author         : lindsey
         date           : 2021/7/20 16:33
-------------------------------------------------
         Change Activity:
             2021/7/20 16:33: todo
-------------------------------------------------
"""
import tensorflow.keras as keras
import tensorflow.keras.datasets.fashion_mnist as fashion
from matplotlib import pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

(x_train, y_train), (x_test, y_test) = fashion.load_data()


# 打印图片
def show(image):
    plt.imshow(image)
    plt.show()


# 将值映射在0-1之间
x_train, x_test = x_train / 255.0, x_test / 255.0

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)

# 扩充训练集，旋转，倒置
image_gen_train = ImageDataGenerator(
    rescale=1. / 1.,  # 二值化
    rotation_range=45,  # 随机旋转45度
    width_shift_range=.15,  # 宽度偏移
    height_shift_range=.15,  # 高度偏移
    horizontal_flip=True,  # 水平旋转
    zoom_range=0.5  # 随机缩放50%
)
image_gen_train.fit(x_train)

model = keras.models.Sequential([
    keras.layers.Conv2D(128, filters=6, kernel_size=(3, 3), strides=1, padding='same'),
    keras.layers.BatchNormalization(),
    keras.layers.Activation('relu'),
    keras.layers.Conv2D(128, filters=6, kernel_size=(3, 3), strides=1, padding='same'),
    keras.layers.MaxPool2D(pool_size=(3, 3), strides=2, padding='same'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])

model.fit(image_gen_train.flow(x_train, y_train, batch_size=32), epochs=5, validation_data=(x_test, y_test))
model.summary()
