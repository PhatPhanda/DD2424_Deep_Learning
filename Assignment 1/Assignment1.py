import pickle 
import numpy as np
from torch_gradient_computations import ComputeGradsWithTorch
import copy

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

def init_parameters(K, d):
    rng = np.random.default_rng()
    # get the BitGenerator used by default_rng
    BitGen = type(rng.bit_generator)
    # use the state from a fresh bit generator
    seed = 42
    rng.bit_generator.state = BitGen(seed).state
    init_net = {}
    init_net['W'] = .01*rng.standard_normal(size = (K, d))
    init_net['b'] = np.zeros((K, 1))

    return init_net, rng



def apply_network(X, network):
    z = np.dot(network['W'],X)
    s = z + network['b']

    P = np.exp(s) / np.sum(np.exp(s), axis=0, keepdims=True)

    return P

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
    reg = lam * np.sum(network['W'] ** 2)
    cost = loss + reg
    return cost

    
def compare_grads(my_grads, torch_grads):
    for key in ['W', 'b']:
        g1 = my_grads[key]
        g2 = torch_grads[key]

        print(f"\nComparing {key}:")
        print("shape my_grads   :", g1.shape)
        print("shape torch_grads:", g2.shape)

        abs_diff = np.abs(g1 - g2)
        max_abs_diff = np.max(abs_diff)

        rel_error = np.max(abs_diff / np.maximum(1e-10, np.abs(g1) + np.abs(g2)))

        print("max absolute difference:", max_abs_diff)
        print("max relative error     :", rel_error)


def mini_batch_GD(X, Y, y, GD_params, init_net, lam, rng):
    trained_net = copy.deepcopy(init_net)
    n = X.shape[1]
    n_batch = GD_params['n_batch']
    eta = GD_params['eta']
    n_epochs = GD_params['n_epochs']

    train_costs = []

    for epoch in range(n_epochs):
        perm = rng.permutation(n)
        X_shuffled = X[:, perm]
        Y_shuffled = Y[:, perm]
    

        for j in range(n // n_batch):
            j_start = j * n_batch
            j_end = (j+1)*n_batch
            Xbatch = X_shuffled[:, j_start:j_end]
            Ybatch = Y_shuffled[:, j_start:j_end]

            P_batch = apply_network(Xbatch, trained_net)
            grads = backward_pass(Xbatch, Ybatch, P_batch, trained_net, lam)

            trained_net['W'] -= eta * grads['W']
            trained_net['b'] -= eta * grads['b']
        
        P_train = apply_network(X, trained_net)
        train_cost = compute_cost(P_train, y, trained_net, lam)
        train_costs.append(train_cost)

        print(f"Epoch {epoch+1}/{n_epochs}, training cost: {train_cost:.6f}")

    return trained_net, train_costs

def test_gradients():
    X_train, Y_train, y_train = load_batch('data_batch_1')
    X_val, Y_val, y_val = load_batch('data_batch_2') 
    X_test, Y_test, y_test = load_batch('data_batch_3')

    X_mean, X_std = compute_stats(X_train)
    X_train = normalize(X_train, X_mean, X_std)
    X_val = normalize(X_val, X_mean, X_std)
    X_test = normalize(X_test, X_mean, X_std)

    d = X_train.shape[0]
    K = Y_train.shape[0]
    
    d_small = 10
    n_small = 3
    lam = 0
    small_net = init_parameters(10,d_small)
    X_small = X_train[0:d_small, 0:n_small]
    Y_small = Y_train[:, 0:n_small]
    P = apply_network(X_small, small_net)
    my_grads = backward_pass(X_small, Y_small, P, small_net, lam)
    torch_grads = ComputeGradsWithTorch(X_small, y_train[0:n_small], small_net)

    
    compare_grads(my_grads, torch_grads)



def main():
    X_train, Y_train, y_train = load_batch('data_batch_1')
    X_val, Y_val, y_val = load_batch('data_batch_2') 
    X_test, Y_test, y_test = load_batch('data_batch_3')

    X_mean, X_std = compute_stats(X_train)
    X_train = normalize(X_train, X_mean, X_std)
    X_val = normalize(X_val, X_mean, X_std)
    X_test = normalize(X_test, X_mean, X_std)
    d = X_train.shape[0]
    K = Y_train.shape[0]
    
    init_net, rng = init_parameters(K, d)
    GD_params = {
    'n_batch': 100,
    'eta': 0.001,
    'n_epochs': 40
    }
    trained_net, train_costs = mini_batch_GD(X_train, Y_train, y_train, GD_params, init_net, 0, rng)
    

main()


