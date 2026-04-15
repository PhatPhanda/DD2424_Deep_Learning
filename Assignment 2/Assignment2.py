import pickle 
import numpy as np
import copy
import matplotlib.pyplot as plt
import torch

def load_batch(filename):
    cifar_dir = 'Assignment 1/Datasets/cifar-10-batches-py/'
    with open(cifar_dir + filename, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    X = (dict[b'data'].astype(np.float64) / 255.0).T
    y = dict[b'labels']
    nn = X.shape[1] 
    Y = np.zeros((10,nn))

    Y[y, np.arange(nn)] = 1

    return X, Y, y

def compute_stats(X):
    mean_X = np.mean(X, axis=1, keepdims=True)
    std_X  = np.std(X, axis=1, keepdims=True)
    return mean_X, std_X

def normalize(X, mean_X, std_X):
    return (X - mean_X) / std_X

def init_parameters(L, d, m, K):
    rng = np.random.default_rng()
    # get the BitGenerator used by default_rng
    BitGen = type(rng.bit_generator)
    # use the state from a fresh bit generator
    seed = 42


    # initialize parameters
    rng.bit_generator.state = BitGen(seed).state
    net_params = {}
    net_params['W'] = [None] * L
    net_params['b'] = [None] * L

    # first layer
    net_params['b'][0] = np.zeros((m,1))
    net_params['W'][0] = 1/np.sqrt(d) * rng.standard_normal((m,d))

    # second layer
    net_params['b'][1] = np.zeros((K,1))   
    net_params['W'][1] = 1/np.sqrt(m) * rng.standard_normal((m,d))

    print(net_params['b'])
    

def apply_network(X, network):

    fp_data = {}
    s1 = np.dot(network['W'][0], X) + network['b'][0]

    # RELU
    h = np.maximum(0, s1)
    s = np.dot(network['W'][1], h) + network['b'][1]

    # SoftMAX
    P = np.exp(s) / np.sum(np.exp(s), axis=0, keepdims=True)

    fp_data['X'] = X
    fp_data['s1'] = s1
    fp_data['h'] = h
    fp_data['s'] = s
    fp_data['P'] = P

    return fp_data

def compute_loss(P, y):
    n = P.shape[1]
    correct_probs = P[y, np.arange(n)]
    L = -np.mean(np.log(correct_probs))
    return L

def compute_accuracy(P, y):
    y_pred = np.argmax(P, axis=0)
    accuracy = np.mean(y_pred == y)

    return accuracy

def backward_pass(X, Y, P, network, lam):
    G = -(Y - P)
    n = X.shape[1]
    grad = {}

    grad['W'] = np.dot(G, X.T) / n + 2*lam*network['W']
    grad['b'] = np.sum(G, axis=1, keepdims=True) / n
    return grad

def compute_cost(P, y, network, lam):
    loss = compute_loss(P, y)
    reg = lam * (torch.sum(torch.multiply(network['W'][0], network['W'][0])) + torch.sum(torch.multiply(network['W'][1], network['W'][1])))
    cost = loss + reg
    return cost

    
def compare_grads(my_grads, torch_grads):
    for key in ['W', 'b']:
        grad1 = my_grads[key]
        grad2 = torch_grads[key]

        print(f"\nComparing {key}:")
        print("shape my_grads   :", grad1.shape)
        print("shape torch_grads:", grad2.shape)

        abs_diff = np.abs(grad1 - grad2)
        max_abs_diff = np.max(abs_diff)

        rel_error = np.max(abs_diff / np.maximum(1e-10, np.abs(grad1) + np.abs(grad2)))

        print("max absolute difference:", max_abs_diff)
        print("max relative error     :", rel_error)

L = 2
m = 50
K = 10
d = 20
init_parameters(L,d,m,K)

     