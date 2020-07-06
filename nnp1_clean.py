#%%
# Load training images

def load_path(path, file_ext = None):
    file_ext = ''
    image_array = []

    image_list = glob(path + '/*' + str(file_ext))  # return path match the pattern

    for img in image_list:
        img = np.asarray(Image.open(img), dtype = float)
        image_array.append(img)
        
    image_array = np.array(image_array).reshape([len(image_list), -1])
    return image_array


dict_train_x = {}
for i in range(10):
    dict_train_x['train_x{}'.format(i)] = load_path("./data/training/{}".format(i), '.png')


dict_train_x.values()
train_x = np.vstack([arr for arr in dict_train_x.values()])


# testing
# path_class_0 = "./data/training/0"
# file_ext = ".png"
# train_x0 = load_path(path_class_0, file_ext)
# train_x0[0].shape
# train_x0[0].dtype
#%%
# Create labels


dict_train_y = {}
len_train_xi = [arr.shape[0] for arr in dict_train_x.values()]

for i in range(10):
    dict_train_y['train_y{}'.format(i)] = np.ones((len_train_xi[i], 1), dtype = float) * i

train_y = np.vstack([arr for arr in dict_train_y.values()])

# testing
# train_y0 = np.zeros((train_x0.shape[0]), dtype = float)*0
# train_y1 = np.ones((train_x1.shape[0]), dtype = float)
# train_y2 = np.ones((train_x2.shape[0]), dtype = float)*2


#%%
# Show images
for j in range(9):
    for i in range(3998, 4002, 1):
        img = np.reshape(train_x[i + j*4000], [28, 28])
        plt.title(int(train_y[i + j*4000]))
        plt.imshow(img, cmap=plt.cm.gray)
        plt.show()


#%%
# Save training data
file = open('./data/train_x.pickle', 'wb')
pickle.dump(train_x, file)
file.close()

file = open('./data/train_y.pickle', 'wb')
pickle.dump(train_y, file)
file.close()

#%%
