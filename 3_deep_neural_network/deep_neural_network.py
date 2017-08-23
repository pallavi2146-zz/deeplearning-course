import numpy as np
from utils import load_dataset, nn_sigmoid, relu, relu_backward, sigmoid_backward

np.random.seed(1)

# Loaded dataset
#-------------------------------------------------------------------------------------
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

# Flatten
train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[1]*train_set_x_orig.shape[2]*train_set_x_orig.shape[3],train_set_x_orig.shape[0])
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[1]*test_set_x_orig.shape[2]*test_set_x_orig.shape[3],test_set_x_orig.shape[0])

# Normalization
train_set_x = train_set_x_flatten/255
test_set_x = test_set_x_flatten/255


# Architecture dimensions
#-------------------------------------------------------------------------------------
layers_dims = [train_set_x.shape[0], 20, 7, 5, 1]


#-------------------------------------------------------------------------------------
def initialize_parameters(layer_dims):
    """

    :param layer_dims:
    :return:
    """
    np.random.seed(3)
    parameters ={}
    for l in range(1,len(layer_dims)):
        parameters["W"+ str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * 0.01
        parameters["b" + str(l)] = np.zeros((layer_dims[l], 1))

    return parameters

#-------------------------------------------------------------------------------------
def linear_forward(A, W, b):

    """


    :return:
    """
    Z = np.dot(W, A) + b
    cache = (A, W, b)

    return Z, cache


def linear_activation_forward(A, W, b, activation):

    if activation == "sigmoid":

        Z, linear_cache = linear_forward(A, W, b)
        A, activation_cache = nn_sigmoid(Z)

    elif activation == "relu":

        Z, linear_cache= linear_forward(A, W, b)
        A, activation_cache = relu(Z)

    cache = (linear_cache, activation_cache)
    return A, cache


def l_model_forward(X ,parameters):

    A = X
    L = len(parameters) // 2
    caches = []

    for l in range(1, L):

        A_prev = A
        A, cache = linear_activation_forward(A_prev, parameters["W"+str(l)], parameters["b"+str(l)], activation="relu")
        caches.append(cache)

    AL, cache = linear_activation_forward(A, parameters["W" + str(L)], parameters["b" + str(L)], activation="sigmoid")
    caches.append(cache)

    return AL, caches

#-------------------------------------------------------------------------------------
def compute_cost(AL, Y):

    m = Y.shape[1]

    cost = (-1/float(m)) * np.sum((Y * np.log(AL) + ((1-Y) * np.log(1-AL))))

    return cost

#-------------------------------------------------------------------------------------
def linear_backward(dZ, cache):

    A_prev, W, b = cache

    m = A_prev.shape[1]

    dW = (1/float(m)) * np.dot(dZ, A_prev.T)
    db = (1/float(m)) * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)

    return dA_prev, dW, db


def linear_activation_backward(dA, cache, activation):

    linear_cache, activation_cache = cache
    dA_prev = dA

    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)


    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    return dA_prev, dW, db


def l_model_backward(AL, Y, caches):
    grads = {}
    L =len(caches)
    m =AL.shape[0]
    Y = Y.reshape(AL.shape)
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

    current_cache = caches[L - 1]
    grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, activation="sigmoid")

    for l in reversed(range(L - 1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 2)], current_cache, activation="relu")
        grads["dA" + str(l + 1)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads


#-------------------------------------------------------------------------------------
def update_parameters(parameters, grads, learning_rate):

    L = len(parameters) / 2

    for l in range(L):
        parameters["W" + str(l + 1)] = parameters['W' + str(l + 1)] - learning_rate * grads["dW" + str(l + 1)]
        parameters["b" + str(l + 1)] = parameters['b' + str(l + 1)] - learning_rate * grads["db" + str(l + 1)]

    return parameters


def L_layer_model(X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost=True):

    np.random.seed(1)
    costs=[]

    parameters = initialize_parameters(layers_dims)
    A = X

    for i in range(0, num_iterations):
        AL, caches =l_model_forward(A, parameters)
        cost = compute_cost(AL, Y)

        grads = l_model_backward(AL, Y, caches)

        parameters = update_parameters(parameters, grads, learning_rate)

        if i % 100 == 0:
            costs.append(cost)
            if print_cost:
                print ("Cost after iteration %i: %f" % (i, cost))

    test_predictions = predict(parameters, test_set_x)
    train_predictions = predict(parameters, train_set_x)

    print (
        'Train Accuracy: %d' % float(
            (np.dot(train_set_y, train_predictions.T) + np.dot(1 - train_set_y, 1 - train_predictions.T)) / float(
                train_set_y.size) * 100) + '%')

    print (
        'Test Accuracy: %d' % float(
            (np.dot(test_set_y, test_predictions.T) + np.dot(1 - test_set_y, 1 - test_predictions.T)) / float(
                test_set_y.size) * 100) + '%')
    return parameters

    return parameters

def predict(parameters, X):

    AL, cache = l_model_forward(X, parameters)
    predictions = (AL > float(0.5)).astype(int)

    return predictions



if __name__=="__main__":
    parameters = L_layer_model(train_set_x, train_set_y, layers_dims, num_iterations=2500, print_cost=True)