# -*- coding: utf-8 -*-
"""
-------------------------------------------------
         File Name      : train-v2-01
         Description    : todo
         Author         : lindsey
         date           : 2021/6/21 0:33
-------------------------------------------------
         Change Activity:
             2021/6/21 0:33: todo
-------------------------------------------------
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
# from sklearn.datasets import load_iris

if __name__ == '__main__':
    model = keras.Sequential()
    model.add(keras.Input(shape=(300, 300, 3)))
    # model.add(layers.Dense(2, activation='relu', input_shape=(300, 300, 3)))
    model.add(layers.Conv2D(32, 5, strides=2, activation='relu'))
    model.add(layers.Conv2D(32, 3, activation='relu'))
    model.add(layers.MaxPool2D(3))
