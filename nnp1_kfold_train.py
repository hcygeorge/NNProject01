#%%
# Working directory
import os
os.getcwd()
os.chdir(os.path.join(os.getcwd(), 'Python_projects/NN_Project01'))

# Built in tools
import pickle
import itertools
from glob import glob  # find file path
import time
from collections import Counter, OrderedDict

# Image processing
from PIL import ImageFont, ImageDraw, Image
import cv2  # 2 means it uses C++ api
print(cv2.__version__)  # 4.1.1

# Data science tools
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy as sp
import seaborn as sns

from sklearn.model_selection import train_test_split
# from sklearn.metrics import precision_score, recall_score, accuracy_score, roc_auc_score
# from imblearn.over_sampling import SMOTENC, RandomOverSampler
#%%
# Create a TwoLayerNet
input_nodes = 784
hidden_nodes = 100
output_nodes = 10

net = TwoLayerNet(input_nodes, hidden_nodes, output_nodes)
#%%
# Tuning hyperparameters

tune = {
    'epochs': 5,
    'batch_size': 32,
    'learing_rate': 0.01}
# todo: early_stop, max_epochs

#%%
# Training loops
def training(train_x, train_y, test_x, test_y, tune_list, verbose = False):
    # Record loss and metrices
    train_loss_list = []
    test_loss_list = []
    train_acc_list = []
    test_acc_list = [] 

    # Train the network
    train_size = train_x.shape[0]
    iter_per_epoch =  max(train_size / tune['batch_size'], 1)

    for e in range(tune['epochs']):
        for i in range(0, train_size, tune['batch_size']):
            batch_x = train_x[i:i+tune['batch_size']]
            batch_y = train_y[i:i+tune['batch_size']]

            # foward except output layer
            y_hat = net.predict(batch_x)

            # compute loss
            loss = net.loss(batch_y, y_hat)  # loss + decay

            # backward
            grad = net.gradient()

            # update weights and bias with SGD
            for key in ('W1', 'b1', 'W2', 'b2'):
                net.params[key] -= tune['learing_rate'] * grad[key]

            # record loss
            # loss = net.loss(batch_x, batch_y)
            train_loss_list.append(loss)

        # compute metrices
        if verbose == False:
            train_acc = net.accuracy(train_x, train_y)
            test_acc = net.accuracy(test_x, test_y)
            train_acc_list.append(train_acc)
            test_acc_list.append(test_acc)
            print("Epoch[%2d/%d], train_acc:%.2f, test_acc:%.2f" % (e+1, tune['epochs'], train_acc, test_acc))
    
    if verbose == False:
        return train_loss_list, train_acc_list, test_acc_list
    else:  # only show the output of last epoch
        train_acc = net.accuracy(train_x, train_y)
        test_acc = net.accuracy(test_x, test_y)
        print("train_acc:%.2f, test_acc:%.2f" % (train_acc, test_acc))
        return train_acc, test_acc



#%%
# Kfold training

# Kfold setting
kf = KFold(n_splits= 10, shuffle= True, random_state = 7)
print(kf)

folds_loss_list = []
folds_acc_list = []
valid_acc_list = [] 

for i, (train_i, test_i) in enumerate(kf.split(train_x)):  # get kfold indices
    folds_x, folds_y = train_x[train_i], train_y[train_i] 
    valid_x, valid_y = train_x[test_i], train_y[test_i]

    start_time = time.time()
    print('Fold[%2d/%d]' % (i+1,10))

    train_loss_list, train_acc_list, test_acc_list = training(train_x, train_y, test_x, test_y, tune)

    folds_loss_list.append(train_loss_list[-1])
    folds_acc_list.append(train_acc_list[-1])
    valid_acc_list.append(test_acc_list[-1])

    print('Processed time:%.2f mins' % ((time.time() - start_time) / 60))

print("Kfold, Train_acc:%.4f, Valid_acc:%.4f" % (np.mean(folds_acc_list), np.mean(valid_acc_list)))
#%%
# Show loss plot
plt.plot(train_loss_list)
plt.plot(train_acc_list)