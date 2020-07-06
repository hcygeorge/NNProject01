#%% 
# Load self-defined tools
from diy_layers import Affine, SoftmaxWithLoss, Sigmoid

#%%
# Define a network
# Weight decay process appears in __init__(), loss() and gradient()
class TwoLayerNet:

    def __init__(self, input_size, hidden_size, output_size, weight_init_std = 0.01, weight_decay_lambda = 0):
        # weight initialization
        self.params = {}
        self.params['W1'] = np.random.normal(0.0, pow(hidden_size, -0.5), (input_size, hidden_size))  # Xavier initialization
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = np.random.normal(0.0, pow(output_size, -0.5), (hidden_size, output_size))
        self.params['b2'] = np.zeros(output_size)

        # Weight decay
        self.weight_decay_lambda = weight_decay_lambda

        # Network structure
        self.layers = OrderedDict()  # dict that keeps order of layers
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Sigmoid1'] = Sigmoid()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
        self.lastLayer = SoftmaxWithLoss()

        # Layers size record (todo)
        # self.hidden_size_list = hidden_size_list
        # self.hidden_layer_num = len(hidden_size_list)

    def predict(self, x):
        """ The whole forward process of network.

        Pass input data through all the forward processes of layers except last layer.
        """
        for layer in self.layers.values():
            x = layer.forward(x)

        return x

    def loss(self, y, y_hat):
        """Conduct predict() and then compute the loss with weight decay."""
        # y_hat = self.predict(x)
        weight_decay = 0

        for i in range(1, 1 + 2):  # self.hidden_layer_num + 2
            W = self.params['W' + str(i)]  # pick up weight params in layer i
            weight_decay += 0.5 * self.weight_decay_lambda * np.sum(W ** 2)  # 0.5 adjusts the constant to 1 after derivation

        return self.lastLayer.forward(y_hat, y) + weight_decay  # loss + decay

    def accuracy(self, x, y):
        """Conduct predict() and then compute the accuracy."""
        y_hat = self.predict(x)  # compute the softmax probs
        y_hat = np.argmax(y_hat, axis=1)  # pick up the position of the max softmax prob as predction of label

        if y.ndim != 1:  # y.dim != 1 means labels are one-hot code
            y = np.argmax(y, axis=1) 

        accuracy = np.sum(y_hat == y) / float(x.shape[0])
        return accuracy

    # def numerical_gradient(self, x, y):
    #     """Compute gradients by numerical derivative, it is recommended to use Back Propagation instead."""
    #     loss_W = lambda W: self.loss(x, y)

    #     grads = {}
    #     grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
    #     grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
    #     grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
    #     grads['b2'] = numerical_gradient(loss_W, self.params['b2'])

    #     return grads

    def gradient(self):
        """Compute the gradients by Back Propagation.
        
        It is actually the whole backward process of network.
        
        Parameters
            x (ndarray): A batch of train_x(features) shape:(batch_size, num_features)
            y (ndarray): A batch of train_y(target) shape:(batch_size, 1)
        
        Return
            grad (dict): A dict of all the updated weight and bias.
        """

        # backward of last layer
        dout = 1  # dloss start as one?
        dout = self.lastLayer.backward(dout)

        # reverse order of layers
        layers = list(self.layers.values())
        layers.reverse()  # "IN PLACE" operation
        
        # Pass loss through all the backward processes of layers in reverse order
        for layer in layers:
            dout = layer.backward(dout)

        # Update the weights and bias
        grads = {}
        grads['W1'] = self.layers['Affine1'].dW + self.weight_decay_lambda * self.layers['Affine1'].W
        grads['b1'] = self.layers['Affine1'].db
        grads['W2'] = self.layers['Affine2'].dW + self.weight_decay_lambda * self.layers['Affine2'].W
        grads['b2'] = self.layers['Affine2'].db
    
        return grads








