
__author__ = "Pallavi Ramicetty"

import numpy as np
from utils import load_dataset, sigmoid

# Loaded dataset
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

# Flatten
train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[1]*train_set_x_orig.shape[2]*train_set_x_orig.shape[3],train_set_x_orig.shape[0])
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[1]*test_set_x_orig.shape[2]*test_set_x_orig.shape[3],test_set_x_orig.shape[0])

# Normalization
train_set_x = train_set_x_flatten/255
test_set_x = test_set_x_flatten/255



def initialize_parameters(dim):

    w = np.zeros((dim,1))
    b = 0

    return w, b


def propagate(w, b, X, Y):

    m = X.shape[1]
    Z = np.dot(w.T, X) + b
    A = sigmoid(Z)
    cost = (-1/float(m)) * np.sum((Y * np.log(A)) + ((1 - Y) * np.log(1 - A)))
    dw = (1/float(m)) * (np.dot(X,(A-Y).T))
    db = (1/float(m)) * (np.sum((A-Y)))

    grads ={
        "dw": dw,
        "db": db
    }

    return grads, cost


def optimize(w, b, X, Y, num_iterations, learning_rate, print_costs = False):

    costs = []
    for i in range(num_iterations):
        grads, cost = propagate(w, b, X, Y)

        dw = grads["dw"]
        db = grads["db"]

        w = w - (learning_rate * dw)
        b = b - (learning_rate * db)

        if i % 100 == 0:
            costs.append(cost)

        if print_costs and i % 100 == 0:
            print ("Cost after iteration %i: %f" % (i, cost))

    params = {"w": w,
              "b": b}

    grads = {"dw": dw,
             "db": db}

    return params, grads, costs


def predict(w, b, X):

    m = X.shape[1]
    w = w.reshape(X.shape[0], 1)

    Z = np.dot(w.T, X) + b
    Y_hat = sigmoid(Z)

    Y_predictions = Y_hat > float(0.5)
    Y_predictions = Y_predictions.astype(int)

    return Y_predictions


def model(train_set_X, train_set_Y, test_set_X, test_set_Y, number_of_iterations = 1000, learning_rate = 0.5, print_costs = False):

    w, b = initialize_parameters(train_set_X.shape[0])
    params, grads, costs = optimize(w, b, train_set_X, train_set_Y, number_of_iterations, learning_rate, print_costs)

    w = params["w"]
    b = params["b"]

    Y_prediction_train = predict(w, b, train_set_X)
    Y_prediction_test = predict(w, b, test_set_X)

    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - train_set_Y)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - test_set_Y)) * 100))

    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test,
         "Y_prediction_train": Y_prediction_train,
         "w": w,
         "b": b,
         "learning_rate": learning_rate,
         "num_iterations": number_of_iterations}

    return d

if __name__ =="__main__":
    d = model(train_set_x, train_set_y, test_set_x, test_set_y, number_of_iterations=10000, learning_rate=0.00005,
              print_costs=True)

