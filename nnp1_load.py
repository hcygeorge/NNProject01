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
# Load training data
with open('./data/train_x.pickle', 'rb') as file:
    Train_x = pickle.load(file)

with open('./data/train_y.pickle', 'rb') as file:
    Train_y = pickle.load(file)

#%%
# Shuffle training data
np.random.seed(7)

Train_xy = np.hstack((Train_x, Train_y))
np.random.shuffle(Train_xy)  # shuffle the rows "IN PLACE"
#%%
# Split training data
seed = 7  # np.random.randint(0, 100000)
train_x, test_x, train_y, test_y = train_test_split(Train_xy[:,0:784], Train_xy[:,784], test_size = 0.2, random_state = seed)

