import pickle 
import numpy as np
import copy
import matplotlib.pyplot as plt
import torch
from torch_gradient_computations import ComputeGradsWithTorch



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

def init_parameters(f, nf, nh, K):
    rng = np.random.default_rng()
    # get the BitGenerator used by default_rng
    BitGen = type(rng.bit_generator)
    # use the state from a fresh bit generator
    seed = 42

    n_p = (32//f)**2
    d0 = n_p * nf 
    d_filter = f * f * 3

    # initialize parameters
    rng.bit_generator.state = BitGen(seed).state
    net_params = {}
    net_params['W'] = [None] * 2
    net_params['b'] = [None] * 2

    net_params['Fs_flat'] = np.sqrt(2 / d_filter) * rng.standard_normal((d_filter, nf))

    # first layer
    net_params['b'][0] = np.zeros((nh, 1))
    net_params['W'][0] = np.sqrt(2/d0) * rng.standard_normal((nh,d0))

    # second layer
    net_params['b'][1] = np.zeros((K,1))   
    net_params['W'][1] = np.sqrt(2/nh) * rng.standard_normal((K,nh))

    return net_params, rng

def create_MX(X, f):
    n = X.shape[1]
    n_p = (32 // f) ** 2

    X_ims = np.transpose(
        X.reshape((32, 32, 3, n), order='F'),
        (1, 0, 2, 3)
    )

    MX = np.zeros((n_p, f*f*3, n))

    for i in range(n):
        l = 0
        for x_cor in range(32 // f):
            for y_cor in range(32 // f):
                x_start = x_cor * f
                x_end = x_start + f

                y_start = y_cor * f
                y_end = y_start + f

                X_patch = X_ims[x_start:x_end, y_start:y_end, :, i]

                MX[l, :, i] = X_patch.reshape((f*f*3,), order='C')

                l += 1

    return MX

def forward_pass(MX, network):
    Fs_flat = network['Fs_flat']
    W1 = network['W'][0]
    b1 = network['b'][0]
    W2 = network['W'][1]
    b2 = network['b'][1]

    n_p = MX.shape[0]
    n = MX.shape[2]
    nf = Fs_flat.shape[1]


    #First Layer
    conv_outputs = np.einsum('ijn, jl ->iln', MX, Fs_flat, optimize=True)

    conv_flat = np.fmax(conv_outputs.reshape((n_p*nf, n), order='C'), 0)

    #ReLu
    x1 = np.maximum(0, W1 @ conv_flat + b1)

    s = W2 @ x1 + b2

    P = np.exp(s) / np.sum(np.exp(s), axis=0, keepdims=True)


    fp_data = {
            'conv_outputs' : conv_outputs,
            'conv_flat' : conv_flat,
            'x1' : x1,
            's' : s,
            'P' : P}


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

def backward_pass(X, Y, fp_data, MX, network):
    Fs_flat = network['Fs_flat']
    W1 = network['W'][0]
    W2 = network['W'][1]

    conv_flat = fp_data['conv_flat']
    x1 = fp_data['x1']
    P = fp_data['P']

    n = X.shape[1]
    n_p = MX.shape[0]
    nf = Fs_flat.shape[1]  

    grads = {} 
    grads['W'] = [None] * 2
    grads['b'] = [None] * 2

    G = -(Y - P)

    # outpute layer
    grads['W'][1] = np.dot(G, x1.T) / n
    grads['b'][1] = np.sum(G, axis=1, keepdims=True) / n

    # hidden layer
    G = W2.T @ G
    G = G * (x1 > 0)    

    grads['W'][0] = (G @ conv_flat.T) / n
    grads['b'][0] = np.mean(G, axis=1, keepdims=True)

    G_batch = W1.T @ G                

    G_batch = G_batch * (conv_flat > 0)

    # Undo flattening
    GG = G_batch.reshape((n_p, nf, n), order='C')

    # Gradient wrt Fs_flat
    MXt = np.transpose(MX, (1, 0, 2))
    grads['Fs_flat'] = np.einsum('ijn, jln -> il', MXt, GG, optimize=True) / n

    return grads    


def compute_cost(P, y, network, lam):
    loss = compute_loss(P, y)
    reg = lam * (np.sum(network['W'][0]**2) + np.sum(network['W'][1]**2))
    cost = loss + reg
    return cost

def compare_grads(my_grads, torch_grads):
    for i in range(2):
        for key in ['W', 'b']:
            grad1 = my_grads[key][i]
            grad2 = torch_grads[key][i]

            print(f"\nComparing {key}{i}:")
            print("shape my_grads   :", grad1.shape)
            print("shape torch_grads:", grad2.shape)

            abs_diff = np.abs(grad1 - grad2)
            max_abs_diff = np.max(abs_diff)

            rel_error = np.max(abs_diff / np.maximum(1e-10, np.abs(grad1) + np.abs(grad2)))

            print("max absolute difference:", max_abs_diff)
            print("max relative error     :", rel_error)

    grad1 = my_grads['Fs_flat']
    grad2 = torch_grads['Fs_flat']
    print(f"\nComparing Fs_flat:")
    print("shape my_grads   :", grad1.shape)
    print("shape torch_grads:", grad2.shape)
    abs_diff = np.abs(grad1 - grad2)
    max_abs_diff = np.max(abs_diff)

    rel_error = np.max(abs_diff / np.maximum(1e-10, np.abs(grad1) + np.abs(grad2)))

    print("max absolute difference:", max_abs_diff)
    print("max relative error     :", rel_error)



def main_exercise1():
    debug_file = 'Assignment 3\debug_info.npz'
    load_data = np.load(debug_file)
    X = load_data['X']
    Fs = load_data['Fs']
    f = Fs.shape[0]
    nf = Fs.shape[3]

    MX = create_MX(X, f)

    Fs_flat = Fs.reshape((f*f*3, nf), order='C')

    network = {'Fs_flat' : Fs_flat,
               'W1' : load_data['W1'],
               'b1' : load_data['b1'],
               'W2' : load_data['W2'],
               'b2' : load_data['b2']

    }

    fp_data = forward_pass(MX, network)

    print(np.allclose(load_data['conv_flat'], fp_data['conv_flat'], atol=1e-10))
    print(np.allclose(load_data['X1'], fp_data['x1'], atol=1e-10))
    print(np.allclose(load_data['P'], fp_data['P'], atol=1e-10))

def main_test_gradients():
    X_train, Y_train, y_train = load_batch('data_batch_1')

    X_mean, X_std = compute_stats(X_train)
    X_train = normalize(X_train, X_mean, X_std)

    f = 4
    nf = 3
    nh = 5
    K = 10
    n_debug = 5


    network, rng = init_parameters(f, nf, nh, K)

    X_small = X_train[:, :n_debug]
    Y_small = Y_train[:, :n_debug]
    y_small = y_train[:n_debug]

    MX = create_MX(X_small, f)

    fp_data = forward_pass(MX, network)
    my_grad = backward_pass(X_small,Y_small, fp_data, MX, network)
    torch_grads = ComputeGradsWithTorch(MX, y_small, network)

    compare_grads(my_grad, torch_grads)





main_test_gradients()