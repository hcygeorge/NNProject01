#%%
from diy_MLP03_backprop import TwoLayerNet
#%%
# Create a TwoLayerNet
input_nodes = 784
hidden_nodes = 256
output_nodes = 10

net = TwoLayerNet(input_nodes,
                  hidden_nodes,
                  output_nodes)

#%%
# Tuning hyperparameters
tune = {
    'epochs': 50,
    'batch_size': 32,
    'learing_rate': 0.01,
    'weight_decay': 0.01} # 0.005
# todo: early_stop, max_epochs

#%%
# Training loops
def training(train_x, train_y, test_x, test_y, tune_list, verbose=False):
    # Record loss and metrices
    train_loss_list = []
    test_loss_list = []
    train_acc_list = []
    test_acc_list = [] 

    # Tune
    net.weight_decay_lambda = tune['weight_decay']
    lr = tune['learing_rate']
    
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
                net.params[key] -= lr * grad[key]

            # record loss
            # loss = net.loss(batch_x, batch_y)
            train_loss_list.append(loss)

        # compute metrices
        train_acc = net.accuracy(train_x, train_y)
        test_acc = net.accuracy(test_x, test_y)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        if verbose == True:
            print("Epoch[%2d/%d], train_acc:%.3f, test_acc:%.3f" % (e+1, tune['epochs'], train_acc, test_acc))
            
        # Adjust parameters
        if (e+1) % 10 == 0:
            lr = lr/2
            print('Learning rate: %.4f' % lr)
    
    if verbose == True:
        return train_loss_list, train_acc_list, test_acc_list
    else:  # only show the output of last epoch
        train_acc = net.accuracy(train_x, train_y)
        test_acc = net.accuracy(test_x, test_y)
        print("train_acc:%.2f, test_acc:%.2f" % (train_acc, test_acc))
        return train_acc, test_acc




#%%
# Train the network
train_loss_list, train_acc_list, test_acc_list = training(train_x, train_y,
                                                          test_x, test_y,
                                                          tune, verbose = True)


#%%
# Show loss plot
plt.plot(train_loss_list)
plt.plot(train_acc_list)
plt.plot(test_acc_list)