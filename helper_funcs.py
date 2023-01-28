import math
import numpy as np
from matplotlib import pyplot as plt


def draw_plot_iris(x_plot, y_plot, df_iris, integer_encd):
    index = ['sepal length in cm', 'sepal width in cm', 'petal length in cm', 'petal width in cm']
    classes = df_iris[4].unique()

    # this formatter will label the colorbar with the correct target names

    plt.figure(figsize=(8, 6))
    plt.scatter(df_iris.loc[:, x_plot], df_iris.loc[:, y_plot], c=integer_encd)

    plt.xlabel(index[x_plot])
    plt.ylabel(index[y_plot])

    plt.tight_layout()
    plt.show()


def accuracy_measure(output, actual):
    class_label = {0: "Iris-setosa", 1: "Iris-versicolor", 2: "Iris-virginica"}

    total_correct = 0
    print("Predicted Output vs Actual Output")
    for i in range(len(output)):
        # print(class_label[np.argmax(output[i])] + " " + class_label[np.argmax(actual[i])])
        if class_label[np.argmax(output[i])] == class_label[np.argmax(actual[i])]:
            total_correct += 1

    print("Total number of instances tested: {}".format(len(output)))
    print("Total number of instances correctly predicted: {}".format(total_correct))
    print("Correctness: {} / {}".format(total_correct, len(output)))
    accuracy = total_correct / len(output) * 100

    return accuracy