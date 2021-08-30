# -*- coding: utf-8 -*-
"""
-------------------------------------------------
         File Name      : predict
         Description    : todo
         Author         : lindsey
         date           : 2021/8/30 14:09
-------------------------------------------------
         Change Activity:
             2021/8/30 14:09: todo
-------------------------------------------------
"""

from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os

# 参数模型
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax")
])

checkpoint_save_path = "./checkpoint/minist.ckpt"
if os.path.exists(checkpoint_save_path + '.index'):
    print('------------load the model---------------')
    model.load_weights(checkpoint_save_path)

img_4 = Image.open("5.png")
img_4 = img_4.resize((28, 28), Image.ANTIALIAS)

plt.imshow(img_4)
plt.show()
img_arr = np.array(img_4.convert('L'))

# 减去背景色
img_arr = 255 - img_arr

# 二值化
img_arr = img_arr / 255.0
x_predict = img_arr[tf.newaxis, ...]


result = model.predict(x_predict)
pred = tf.argmax(result, axis=1)
tf.print(pred)
