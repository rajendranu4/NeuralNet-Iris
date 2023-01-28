import numpy as np


class NeuralNetworkIris:
    def __init__(self, nn_x, nn_h, nn_y):
        # weights for hidden layer, output layer and bias for the hidden layer and output layer
        # learning rate is defaulted to 0.01
        self.weights_h = np.random.randn(nn_x, nn_h)
        self.weights_o = np.random.randn(nn_h, nn_y)
        self.bias_h = np.zeros((1, nn_h))
        self.bias_o = np.zeros((1, nn_y))
        self.learning_rate = 0.01

    def set_hyperparameters(self, learning_rate):
        self.learning_rate = learning_rate

    def get_parameters(self):
        return np.round_(self.weights_h, decimals=4), np.round_(self.weights_o, decimals=4), \
               np.round_(self.bias_h, decimals=4), np.round_(self.bias_o, decimals=4)

    def sigmoid(self, X):
        return 1 / (1 + np.exp(-X))

    def derivative_sigmoid(self, X):
        return self.sigmoid(X) * (1 - self.sigmoid(X))

    def softmax(self, X):
        return np.exp(X) / np.sum(np.exp(X), axis=1, keepdims=True)

    def cost_cross_entropy(self, activation_o, y):
        return np.sum(-(y * np.log(activation_o)))

    def feed_forward(self, X):

        # (inputs * weights for hidden layer) + bias of hidden layer
        hidden = X.dot(self.weights_h) + self.bias_h

        # sigmoid function is the activation function for hidden layer output
        activation_h = self.sigmoid(hidden)

        # (hidden layer activated outputs * weights for output layer) + bias of output layer
        output = activation_h.dot(self.weights_o) + self.bias_o

        # softmax function is the activation function for ouptut layer since
        # probability distribution is required for the 3 classes for classification
        # the class labels are one hot encoded [1, 0, 0], [0, 1, 0] and [0, 0, 1] for the
        # 3 classes of flowers Iris-setosa, Iris-versicolor and Iris-virginica respectively
        activation_o = self.softmax(output)

        return hidden, activation_h, activation_o

    def backpropagation(self, hidden, activation_o, activation_h, X, y):

        # calculating margin_of_error = predicted - actual
        o_error = y - activation_o

        # calculting output_delta = margin_of_error * derivative sigmoid of output activation
        o_delta = o_error * self.derivative_sigmoid(activation_o)

        # weight and bias adjustments for output layer
        dwo = (activation_h.T).dot(o_delta)
        dbo = np.sum(o_delta, axis=0, keepdims=True)

        # calculating hidden layer error from output_delta
        h_error = o_delta.dot(self.weights_o.T)
        h_delta = h_error * self.derivative_sigmoid(activation_h)

        # weight and bias adjustments for output layer
        dwh = (X.T).dot(h_delta)
        dbh = np.sum(h_delta, axis=0, keepdims=True)

        # adjustments ==> new weight or bias = existing weight or bias + (learning rate * weight or bias adjustment)
        self.weights_o += (self.learning_rate * dwo)
        self.weights_h += (self.learning_rate * dwh)
        self.bias_o += (self.learning_rate * dbo)
        self.bias_h += (self.learning_rate * dbh)

    def train(self, X, y, epochs):
        if epochs <= 1000:
            acc_step = 10
        else:
            acc_step = 100

        # epochs - number of iterations the train dataset will go through and the weights and bias of
        # hidden and output layers are adjusted during each iteration
        for i in range(0, epochs):
            hidden, activation_h, activation_o = self.feed_forward(X)
            '''print("validating")
            print("original class {}".format(y[0]))
            print("predicted class {}".format(np.round_(activation_o[0], decimals=2)))'''
            self.backpropagation(hidden, activation_o, activation_h, X, y)

            if i % acc_step == 0:
                print("Loss on epoch {} ===> {}".format(i, self.cost_cross_entropy(activation_o, y)))

    def predict(self, X):

        # similar to feedforward with the inputs X
        # the weights and bias of the hidden and output layers are calculated during the training
        # and are updated in the respective class variables during each iteration of the training
        # The current weights and bias will be used to predict the class of unseen data
        hidden = X.dot(self.weights_h) + self.bias_h
        activation_h = self.sigmoid(hidden)
        output = activation_h.dot(self.weights_o) + self.bias_o
        activation_o = np.exp(output) / np.sum(np.exp(output), axis=1, keepdims=True)

        return np.round_(activation_o, decimals=2)