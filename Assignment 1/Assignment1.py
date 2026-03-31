import pickle 
import numpy as np

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

    return init_net



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
    lam = 0.2
    init_net = init_parameters(K, d)

    P = apply_network(X_train[:, 0:100], init_net)
    print(backward_pass(X_train[:, 0:100], Y_train[:, 0:100],P, init_net, lam))
    



main()







