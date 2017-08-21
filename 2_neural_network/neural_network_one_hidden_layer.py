import numpy as np
from utils import load_dataset, sigmoid

# Reference - http://cs231n.github.io/neural-networks-case-study/

# Loaded dataset
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

# Flatten
train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[1]*train_set_x_orig.shape[2]*train_set_x_orig.shape[3],train_set_x_orig.shape[0])
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[1]*test_set_x_orig.shape[2]*test_set_x_orig.shape[3],test_set_x_orig.shape[0])

# Normalization
train_set_x = train_set_x_flatten/255
test_set_x = test_set_x_flatten/255

# initialized parameters

def initialize_parameters(n_x, n_h, n_y):

    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h,1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y,1))

    parameters = {
        "W1": W1,
        "b1": b1,
        "W2": W2,
        "b2": b2,

    }

    return parameters


def forward_propagation(X, parameters):

    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    Z1 = np.dot(W1, X) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)

    cache ={

        "Z1": Z1,
        "A1": A1,
        "Z2": Z2,
        "A2": A2

    }

    return A2, cache


def compute_cost(A2, Y):

    m = Y.shape[1]

    cost = (-1/float(m)) * np.sum ((Y * np.log(A2) + ((1-Y) * np.log(1-A2))))

    return cost


def backward_propagation(X, Y, cache, parameters):

    m = X.shape[1]

    A1 = cache["A1"]
    A2 = cache["A2"]

    W2 = parameters["W2"]

    dZ2 = A2 - Y
    dW2 = (1/float(m)) * np.dot(dZ2, A1.T)
    db2 = (1/float(m)) * np.sum(dZ2, axis = 1, keepdims = True)
    dZ1 = np.dot(W2.T, dZ2) * (1 - np.power(A1, 2))
    dW1 = (1 / float(m)) * np.dot(dZ1, X.T)
    db1 = (1 / float(m)) * np.sum(dZ1, axis=1, keepdims=True)

    grads = {
        "dW1":dW1,
        "db1":db1,
        "dW2":dW2,
        "db2":db2
    }

    return grads

def update_parameters(parameters, grads, learning_rate):

    W1 = parameters["W1"] - (learning_rate * grads["dW1"])
    b1 = parameters["b1"] - (learning_rate * grads["db1"])
    W2 = parameters["W2"] - (learning_rate * grads["dW2"])
    b2 = parameters["b2"] - (learning_rate * grads["db2"])

    parameters = {
        "W1": W1,
        "b1": b1,
        "W2": W2,
        "b2": b2,

    }

    return parameters


def model(X, Y, n_h, num_iterations = 10000, print_cost= True, learning_rate= 00.2):

    n_x = X.shape[0]
    n_y = Y.shape[0]

    parameters = initialize_parameters(n_x, n_h, n_y)
    costs = []
    for i in range(0,num_iterations):
        A2, cache = forward_propagation(X, parameters)
        cost = compute_cost(A2, Y)
        costs.append(cost)
        grads = backward_propagation(X, Y, cache, parameters)
        parameters = update_parameters(parameters, grads, learning_rate)
        if print_cost and (i % 100) ==0:
            print ("Cost after iteration %i: %f" % (i, cost))

    test_predictions = predict(parameters, test_set_x)
    train_predictions = predict(parameters, train_set_x)

    print (
        'Train Accuracy: %d' % float(
            (np.dot(test_set_y, train_predictions.T) + np.dot(1 - test_set_y, 1 - train_predictions.T)) / float(
                test_set_y.size) * 100) + '%')

    print (
        'Accuracy: %d' % float(
            (np.dot(test_set_y, test_predictions.T) + np.dot(1 - test_set_y, 1 - test_predictions.T)) / float(
                test_set_y.size) * 100) + '%')
    return parameters

def predict(parameters, X):

    A2, cache = forward_propagation(X, parameters)
    predictions = (A2 > float(0.5)).astype(int)

    return predictions


if __name__=="__main__":

    parameters = model(train_set_x, train_set_y, n_h=4, num_iterations=10000, print_cost=True)






