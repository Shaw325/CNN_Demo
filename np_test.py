# -*- coding: utf-8 -*-
"""
-------------------------------------------------
         File Name      : np_test
         Description    : todo
         Author         : lindsey
         date           : 2021/7/20 15:35
-------------------------------------------------
         Change Activity:
             2021/7/20 15:35: todo
-------------------------------------------------
"""

import numpy as np

original_a = [(1, 2, 3), (4, 5, 6)]
b = np.asarray([[[1, 2, 3], [3, 4, 5]], [[3, 1, 5], [312, 543, 1]]])
a = np.asarray(original_a)
a = np.hstack(a)
print(a)
print(b)
