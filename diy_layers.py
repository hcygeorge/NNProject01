#%% [markdown]
# 實現神經網路的各種計算層
# 乘法層
# 加法層
# 啟動層
# 仿射層
# SoftmaxWithLoss
# Batch Normalization

# 每一種層都有自己的正向和反向計算過程

#%%
import numpy as np

#%%
# 乘法層
# Multiplication 
class MulLayer():
    def __init__(self):
        self.x = None
        self.y = None
    
    def forward(self, x, y):  # input, weight
        self.x = x
        self.y = y
        out = x * y

        return out

    def backward(self, dout):
        dx = dout * self.y  
        dy = dout * self.x

        return dx, dy

# testing
# 假設費用為一個蘋果價格*個數*(1+稅率)
# 反向過程:根據乘法微分，誤差對一個變數微分的結果為誤差乘上另一個變數
# apple = 100
# num_apple = 2
# tax = 1.1

# mul_apple_layer = MulLayer()
# mul_tax_layer = MulLayer()

# apple_price = mul_apple_layer.forward(apple, num_apple)  # 蘋果價格*個數
# price = mul_tax_layer.forward(apple_price, tax)  # 蘋果價格*個數*(1+稅率)
# print(price)
# dprice = 1
# dapple_price, dtax = mul_tax_layer.backward(dprice)
# dapple, num_dapple = mul_apple_layer.backward(dapple_price)
# print(dapple, num_dapple, dtax)  # 2.2 110 200
#%%
# Addition
class AddLayer():
    def __init__(self):
        pass
    
    def forward(self, x, y):
        out = x + y  # input, bias

        return out
    
    def backward(self, dout):
        dx = dout * 1
        dy = dout * 1
        return dx, dy


#%%
# Relu
class Relu:
    def __init__(self):
        self.mask = None
    
    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0

        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout

        return dx

# testing
# x = np.array([[-1.0, -0.5], [-2.0, 3.0]])
# print(x)

# mask = (x <=0)
# print(mask)

#%%
# Sigmoid
# 計算已經組合成一個節點
class Sigmoid:
    def __init__(self):
        self.out = None
    
    def forward(self, x):
        # print(x.shape)  # to debug
        # x = x - np.max(x, axis=0)  # each output substracts it maximum to avoid overflow values
        out = 1 / (1 + np.exp(-x))
        self.out = out

        return out
    
    def backward(self, dout=1):
        dx = dout * (1 - self.out) * self.out  # dfx = fx*(1-fx)
        
        return dx

class SigmoidWithLoss:
    def __init__(self):
        self.y_hat = None
        self.y = None
        self.loss = None

    def forward(self, x, y):
        y_hat = 1 / (1 + np.exp(-x))
        self.y_hat = y_hat
        self.y = y
        self.loss = self.y_hat - self.y

        return self.loss
    
    def backward(self, dout=1):
        dx = dout * (1 - self.out) * self.out
        
        return dx

#%%
# Affine(fully connected layer)
# 仿射變換為兩函數的複合：平移及線性映射
# x*w為線性映射的純量乘法，x*w + b為平移
class Affine:
    """A fully connetcted layer of neural network for multi-channels input."""
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.original_x_shape = None  # hold multi-channels shape
        self.dW = None
        self.db = None
    
    def forward(self, x):
        """Conduct dot product for input data and weights and then a bias is added on.
        
        Parameters
            x(ndarray): A matrix of input data.
                The shape of input x should be (batch, channels, height, weight)
        
        Returns
            out(float): A matrix of output data with shape (batch, num_all_pixels)
        """
        self.original_x_shape = x.shape  # hold matrix shape for recovery
        # print(x.shape)  # to debug
        x = x.reshape(x.shape[0], -1)  # flatten multi-channels (batch, num_all_channels_pixels)
        self.x = x
        out = np.dot(self.x, self.W) + self.b  # x*W + b

        return out

    def backward(self, dout):
        """Conduct back-propagation."""
        dx = np.dot(dout, self.W.T)  # dloss/dx = loss*W 
        self.dW = np.dot(self.x.T , dout)
        self.db = np.sum(dout, axis = 0)  # dloss/db = loss*W 

        dx = dx.reshape(*self.original_x_shape)  # recover dx to multi-channels
        return dx

#%%
# Softmax and CrossEntropy

class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y_hat = None
        self.y = None

    # Softmax 多分類問題損失函數
    def softmax(self, x):
        """
        Args:
            x  (ndarray): (batch, num_all_pixels)
            y_hat (ndarray): Softmax outputs from the network.
        """
        if x.ndim == 2:
            x = x.T 
            x = x - np.max(x, axis=0)  # each output substracts it maximum to avoid overflow values
            y_hat = np.exp(x) / np.sum(np.exp(x), axis=0)
            return y_hat.T  # (num_class, batch)

        x = x - np.max(x)
        y_hat = np.exp(x) / np.sum(np.exp(x))
        return y_hat  # (batch, num_class) send to self.yhat 

    # CrossEntropy:給定一個分類策略，猜中集合內任一元素的分類所需的期望次數(越小越好)
    def cross_entropy(self, y, y_hat):

        if y_hat.ndim == 1:
            y = y.reshape(1, y.size)
            y_hat = y_hat.reshape(1, y_hat.size) 

        # if ground truths are one-hot code
        if y_hat.size == y.size:
            y = y.argmax(axis=1)

        batch_size = y_hat.shape[0]
        # when y are single labels like "3"
        return -np.sum(np.log(y_hat[np.arange(batch_size), y.astype(int)] + 1e-7)) / batch_size  # array as indices should be type int
    
    def forward(self, x, y):
        self.y = y
        self.y_hat = self.softmax(x)
        self.loss = self.cross_entropy(self.y, self.y_hat)

        return self.loss
    
    def backward(self, dout=1): 
        batch_size = self.y.shape[0]
        dx = self.y_hat.copy() 
        # Derivative of Cross Entropy Loss with Softmax
        dx[np.arange(batch_size), self.y.astype(int)] -= 1  # dL/dout = y_hat - y
        dx = dx / batch_size  # 

        return dx

        # if self.y.size == self.y_hat.size:  # when label is an one-hot vector
        #     dx = (self.y_hat - self.y) / batch_size
        # else:
            # dx = self.y_hat.copy()
            # dx[np.arange(batch_size), self.y] -= 1
            # dx = dx / batch_size
# testing
# y_hat = np.random.uniform(0, 1, size = [32, 10])  # (batch, num_class)
# y = np.random.randint(0, 10, size = [32, 1])  # (batch, 1)
# gar01 = SoftmaxWithLoss()
# gar02 = gar01.cross_entropy(y, y_hat)  # float
#%%
# Batch Normalization
# Normalize batch data (output of Affine or Conv) before activation
class BatchNormalization:
    def __init__(self, gamma, beta, momentum=0.1, running_mean=None, running_var=None):
        self.gamma = gamma  # scale after normalization
        self.beta = beta  # shift after normalization
        self.momentum = momentum  # smoothing factor of EMA, momentum * newest value
        # temporary variables in forward()
        self.input_shape = None  # Conv:4dim, Affine:2dim
        # EMA of batch mean and var, used while testing 
        self.running_mean = running_mean
        self.running_var = running_var
        # temporary variables in backward()
        self.batch_size = None
        self.xc = None  # centered batch data
        self.std = None
        self.dgamma = None
        self.dbeta = None
    
    def forward(self, x, train_flg=True):
        self.input_shape = x.shape  # hold original shape here
        if x.ndim != 2:  # If the previous layer is not Affine(ndim=2)
            N, C, H, W = x.shape
            x = x.reshape(N, -1)

        out = self.__forward(x, train_flg)  # x.ndim==2 at this moment

        return out.reshape(*self.input_shape)  # recover the shape of x after normalization

    def __forward(self, x, train_flg):
        """The actual forward process of batch normalization."""
        if self.running_mean is None:
            N, D = x.shape
            # In the first iteration,setting moving average and var as 0.
            self.running_mean = np.zeros(D)
            self.running_var = np.zeros(D)

        if train_flg:
            mu = x.mean(axis=0)
            xc = x - mu
            var = np.mean(xc**2, axis=0)
            std = np.sqrt(var + 10e-7)
            xn = xc / std

            self.batch_size = x.shape[0]
            self.xc = xc
            self.xn = xn
            self.std = std
            self.running_mean = (1-self.momentum) * self.running_mean + self.momentum * mu
            self.running_var = (1-self.momentum) * self.running_var + self.momentum * var
        else:
            xc = x - self.running_mean
            xn = xc / ((np.sqrt(self.running_var + 10e-7)))

        out = self.gamma * xn + self.beta
        return out

    def backward(self, dout):
        if dout.ndim != 2:
            N, C, H, W = dout.shape
            dout = dout.reshape(N, -1)
        
        dx = self.__backward(dout)

        return dx.reshape(*self.input_shape)

    def __backward(self, dout):
        dbeta = dout.sum(axis=0)
        dgamma = np.sum(self.xn * dout, axis=0)
        dxn = self.gamma * dout
        dxc = dxn / self.std
        dstd = -np.sum((dxn * self.xc) / (self.std * self.std), axis=0)
        dvar = 0.5 * dstd / self.std
        dxc += (2.0 / self.batch_size) * self.xc * dvar
        dmu = np.sum(dxc, axis=0)
        dx = dxc - dmu / self.batch_size

        self.dgamma = dgamma
        self.dbeta = dbeta

        return dx

# testing
# x01 = np.random.randint(0,10,(3, 3))
# x01.sum(axis=0)
#%%
# Dropout
# The effect of dropout is somewhat like ensemble learning
# because it use differnet nodes of network in each iteration of training.
class Dropout:
    """http://arxiv.org/abs/1207.0580"""
    def __init__(self, dropout_ratio=0.5):
        self.dropout_ratio = dropout_ratio
        self.mask = None
    
    def forward(self, x, train_flg=True):
        if train_flg:
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio
            return x * self.mask
        else:
            return x * (1.0 - self.dropout_ratio)
    def backward(self, dout):
        return dout * self.mask  # If signal pass in forward,it pass in backward too.

#%%
# Convolution
# backward process is in util.col2im()
class Convolution:
    def __init__(self, W, b, stride=1, pad=0):
        self.W = W  # filter weights
        self.b = b  # filter biases
        self.stride = stride
        self.pad = pad

        # attributes used in backward
        self.x = None
        self.col = None
        self.col_W = None

        # gradients of W and b
        self.dW = None
        self.db = None

    def forward(self, x):
        FN, C, FH, FW = self.W.shape  # FN:number of filters
        N, C, H, W = x.shape
        out_h = int((H + 2*self.pad - FH) / self.stride) + 1
        out_w = int((W + 2*sel.pad - FW) / self.stride) + 1

        col = im2col(x, FH, FW, self.stride, self.pad)  # flatten input data
        col_W = self.W.reshape(FN, -1).T  # flatten filter
        
        out = np.dot(col, col_W) + self.b
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)  # (N,H,W,C)->(N,C,H,W)

        self.x = x
        self.col = col
        self.col_W = col_W

        return out

    def backward(self, dout):
        FN, C, FH, FW = self.W.shape
        dout = dout.transpose(0, 2, 3, 1).reshape(-1, FN)

        self.db = np.sum(dout, axis=0)
        self.dW = np.dot(self.col.T, dout)
        self.dW = self.dW.transpose(1, 0).reshape(FN, C, FH, FW)

        dcol = np.dot(dout, self.col_W.T)
        dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)

        return dx

        