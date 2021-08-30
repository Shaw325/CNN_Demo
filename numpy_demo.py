# -*- coding: utf-8 -*-
"""
-------------------------------------------------
         File Name      : numpy_demo
         Description    : todo
         Author         : lindsey
         date           : 2021/6/28 1:23
-------------------------------------------------
         Change Activity:
             2021/6/28 1:23: todo
-------------------------------------------------
"""

import numpy as np

if __name__ == '__main__':
    x = np.arange(10)
    y = np.arange(10)
    np.random.seed(10)
    np.random.shuffle(x)
    np.random.seed(10)
    np.random.shuffle(y)
    print(x,y)
