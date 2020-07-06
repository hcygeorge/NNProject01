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
#%%
path = 'E:/testing'

# get list of path
image_list = glob(path + '/*' + '.png')  # return path match the pattern

# convert to array
image_array = []
for img in image_list:
    img = np.asarray(Image.open(img), dtype = float)
    image_array.append(img)
    
tests = np.array(image_array).reshape([len(image_list), -1])

#%%
# extract the labels from the path
ind = []

for path in image_list:
    ind.append(path[-9:-4])


#%%
