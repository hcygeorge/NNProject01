#%% [markdown]
# 實現神經網路的額外工具
# im2col: 將輸入資料(圖像)和卷積核展開成向量排列
#%%
import numpy as np
#%%
# im2col
# 為了避免使用多重迴圈執行卷積運算(無效率)
# 將輸入資料和filter展開成向量，用矩陣乘法完成卷積運算

def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    N, C, H, W = input_data.shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1

    img = np.pad(input_data, [(0,0), (0,0), (pad, pad), (pad, pad)], 'constant')  # 'constant' pads with 0
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

    # 以下雙重迴圈目的是展開每個卷積核元素對應到的輸入資料範圍
    # 因此每一次被展開的輸入資料的區塊大小是卷積核滑動的距離(out_h, out_w)而非卷積核的長寬
    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]  # img[...].shape = (:, :, out_h, out_w)

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1)
    return col 

# testing im2col
# img = np.random.randn(32, 3, 7, 7)
# col = np.zeros((32, 3, 5, 5, 3, 3))
# for y in range(5):
#     y_max = y + 1*3
#     for x in range(5):
#         x_max = x + 1*3
#         col[:, :, y, x, :, :] = img[:, :, y:y_max:1, x:x_max:1]
# col.shape  # (32, 3, 5, 5, 3, 3)
# col02 = col.transpose(0, 4, 5, 1, 2, 3)  # (N, out_h, out_w, channel, filter_h, filter_w)
# col03 = col02.reshape(32*3*3, -1)  #   (N*out_h*out_w, channel*filter_h*filter_w)
# # col02 的每一行對應到同一個卷稽核元素，矩陣乘法時卷稽核的元素展開成列
# # 也代表展開後的卷稽核參數有75個
# print(col02.shape, col03.shape)  # (32, 3, 3, 3, 5, 5) (288, 75)
# #
# col04 = im2col(img, 5, 5, 1, 0)
# print(col04.shape)  # (288, 75)
# test np.pad
# input_data = np.random.randn(32, 3, 28, 28)
# pad = 1
# img = np.pad(input_data, [(0,0), (0,0), (pad, pad), (pad, pad)], 'constant')  # 'constant' pads with 0
# img.shape  # (32, 3, 30, 30)


#%%
# col2im
def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    """Turn columns of data into a batch of images.

    This function is used in backward() of Convoluiton class.

    Parameters
        col (ndarray): Columns of data.
        input_shape (): Original shape of one batch of images.(ex:(32,3,28,28))
            This is the only parameter col2im different from im2col.
        filter_h
        filter_w
        stride
        pad
    
    Returns
    """
    N, C, H, W = input_shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2(pad - filter_w)//stride + 1
    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)
    # recover col.transpose(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1)

    img = np.zeros((N, C, H + 2*pad + stride - 1, W + 2*pad + stride - 1))  # 2*pad+stride-1 is duplicate values from im2col
    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]  # img includes duplicate values

    return img[:, :, pad:H + pad, pad:W + pad]  # (N, C, H, W)