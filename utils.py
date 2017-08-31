
__author__ = "Pallavi Ramicetty"

import h5py
import numpy as np
import os
import sys


def load_dataset():

    train_dataset = h5py.File("datasets/train_catvnoncat.h5", "r")
    test_dataset = h5py.File("datasets/test_catvnoncat.h5", "r")

    train_set_x_orig = np.array(train_dataset["train_set_x"][:])
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))

    test_set_x_orig = np.array(test_dataset["test_set_x"][:])
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    classes = np.array(test_dataset["list_classes"][:])

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


def sigmoid(z):

    A = 1 / (1 + np.exp(-z))

    return A

def nn_sigmoid(z):

    A = 1 / (1 + np.exp(-z))
    cache = z

    return A, cache


def relu(z):

    A = np.maximum(0, z)
    cache = z

    return A, cache
def relu_backward(dA, cache):

    Z = cache
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0

    return dZ

def sigmoid_backward(dA, cache):

    Z = cache

    s = 1 / (1 + np.exp(-Z))
    dZ = dA * s * (1 - s)

    assert (dZ.shape == Z.shape)

    return dZ
