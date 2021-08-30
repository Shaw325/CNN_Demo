# -*- coding: utf-8 -*-
"""
-------------------------------------------------
         File Name      : keras_iris
         Description    : todo
         Author         : lindsey
         date           : 2021/7/4 15:38
-------------------------------------------------
         Change Activity:
             2021/7/4 15:38: todo
-------------------------------------------------
"""

import tensorflow.keras as keras
from sklearn.datasets import load_iris
import numpy as np
import os


def load_data():
    # 加载150条鸢尾花数据
    x_data = load_iris().data
    y_data = load_iris().target
    return x_data, y_data


x_data, y_data = load_data()

np.random.seed(116)
np.random.shuffle(x_data)
np.random.seed(116)
np.random.shuffle(y_data)

model = keras.models.Sequential([
    keras.layers.Dense(3, activation='softmax', kernel_regularizer=keras.regularizers.l2())]
)

model.compile(optimizer=keras.optimizers.SGD(learning_rate=0.1),
              loss='sparse_categorical_crossentropy',
              metrics=['sparse_categorical_accuracy'])

train_model_save_path = "./trainModel/iris.ckpt"
if os.path.exists(train_model_save_path + ".index"):
    print("=============== load the model==================")
    model.load_weights(train_model_save_path)

cp_callback = keras.callbacks.ModelCheckpoint(filepath=train_model_save_path,
                                              save_weights_only=True,
                                              monitor='val_loss',
                                              save_best_only=True)
model.fit(x_data, y_data, epochs=500, validation_freq=20, validation_split=0.2, callbacks=cp_callback)
model.summary()
