# -*- coding: utf-8 -*-
"""
-------------------------------------------------
         File Name      : graph
         Description    : todo
         Author         : lindsey
         date           : 2021/8/12 10:03
-------------------------------------------------
         Change Activity:
             2021/8/12 10:03: todo
-------------------------------------------------
"""
import matplotlib.pyplot as plt
import numpy as np

r = 1
eta = np.arange(0, 2 * np.pi, 0.01)
x = r * np.sin(eta)
y =  np.sqrt(1 - np.power(x, 2))/r

plt.plot(x, y)
plt.title("xx")
plt.show()
