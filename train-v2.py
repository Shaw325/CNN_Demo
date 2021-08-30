# -*- coding: utf-8 -*-
"""
-------------------------------------------------
         File Name      : train-v2
         Description    : todo
         Author         : lindsey
         date           : 2021/6/21 0:16
-------------------------------------------------
         Change Activity:
             2021/6/21 0:16: todo
-------------------------------------------------
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

if __name__ == '__main__':
    model = keras.Sequential(
        [
            layers.Dense(2, activation="relu", name="layer1"),
            layers.Dense(3, activation="relu", name="layer2"),
            layers.Dense(4, name="layer3"),
        ]
    )
    x = tf.ones((3, 3))
    y = model(x)
    print(x, y)
