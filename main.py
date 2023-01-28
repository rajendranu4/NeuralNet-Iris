import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

from nn_iris import NeuralNetworkIris
import helper_funcs as helpers


if __name__ == '__main__':
    # loading iris dataset
    df_iris = pd.read_csv('iris.data', header=None)


    hidden_neurons = 6  # number of neurons in the hidden layer
    epochs = 1000      # number of iterations the train dataset will go through for training

    # preparing input and output values for training and testing split
    df_array = df_iris.values
    X = df_array[:,0:4].astype(float)
    y = df_array[:,4]

    # converting the class labels Iris-setosa, Iris-versicolor and Iris-virginica to one-hot encoded labels
    label_encr = LabelEncoder()
    onehot_encr = OneHotEncoder(sparse=False)
    integer_encd = label_encr.fit_transform(y)

    integer_encd = integer_encd.reshape(len(integer_encd), 1)
    onehot_encd = onehot_encr.fit_transform(integer_encd)

    print(onehot_encd)

    # splitting the dataset into training set and testing set
    X_train, X_test, y_train, y_test = train_test_split(X, onehot_encd, train_size = 0.8, shuffle = True)

    helpers.draw_plot_iris(0, 1, df_iris, integer_encd) # 0 vs 1 columns in df_iris dataset

    helpers.draw_plot_iris(2, 3, df_iris, integer_encd)  # 2 vs 3 columns in df_iris dataset

    # invoking the neural network class created
    network_iris = NeuralNetworkIris(X_train.shape[1], hidden_neurons, y_train.shape[1])

    weights_h, weights_o, bias_h, bias_o = network_iris.get_parameters()

    print("Parameters before training: ")
    print("Hidden Layer Weights:")
    print(weights_h)
    print("Output Layer Weights:")
    print(weights_o)
    print("Hidden Layer Bias:")
    print(bias_h)
    print("Output Layer Bias:")
    print(bias_o)

    # training inputs X_train is passed to train() of the iris neural network class
    # Feed forward and backpropation are done for the number of epochs mentioned
    network_iris.train(X_train, y_train, epochs)

    weights_h, weights_o, bias_h, bias_o = network_iris.get_parameters()

    print("Parameters after training: ")
    print("Hidden Layer Weights:")
    print(weights_h)
    print("Output Layer Weights:")
    print(weights_o)
    print("Hidden Layer Bias:")
    print(bias_h)
    print("Output Layer Bias:")
    print(bias_o)

    # testing inputs X_test is passed predict() and the output will be
    output = network_iris.predict(X_test)
    accuracy = helpers.accuracy_measure(output, y_test)

    print("Accuracy: {}".format(accuracy))