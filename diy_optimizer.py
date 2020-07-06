#%%
import numpy as np
import matplotlib.pyplot as plt

#%%
# SGD
class SGD:
    def __init__(self, lr = 0.01):
        self.lr = lr
    def update(self, params, grads):
        for key in params.keys():
            params[key] -= self.lr * grads[key]  # W* = W - lr*gradient

#%%
# Momentum
# 計算梯度時會保留前幾次迭代的梯度
# 即使當次的梯度為0，之前迭代的梯度會繼續更新權重，有利於跳脫局部極值
# 引入速度變數v更新權重，代表權重更新的速度而梯度被用於更新速度
# 換句話說梯度更新的是權重更新的速度而不是更新權重本身
# 速度的初始值為0，第一次迭代時Momentum跟一般SGD一樣
# momentum習慣設為0.9，代表上一次迭代的梯度會保留9成變成下次迭代的速度
# 有些做法會在梯度前乘上1-momentum，但可被省略
class Momentum:
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = monmentum
        self.v = None  

    def update(self, params, grads):
        if self.v is None:
            self.v = {}
            for key, val in params.items():
                self.v[key] = np.zeros_like(val)
            
            for key in params.keys():
                self.v[key] = self.momentum * self.v[key] + self.lr * grads[key]
                params[key] -= self.v[key]
    
#%%
# AdaGrad
# learning rate decay
# 引入變數h累積每次梯度的平方，成為過去梯度的平方和
# 每次迭代時lr除以h^0.5，調整學習速度
# 缺點 更新幅度最後會趨近0

class AdaGrad:
    def __init__(self, lr=0.01):
        self.lr = lr
        self.h = None

    def update(self, params, grads):
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)
    
        for key in params.keys():
            self.h[key] += grads[key] * grads[key]
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7)  # Add 1e-7 avoid divide by 0

