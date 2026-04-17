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
    net_params['W'][1] = 1/np.sqrt(m) * rng.standard_normal((K,m))

    return net_params, rng
    

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

def backward_pass(X, Y, fp_data, network, lam):
    P = fp_data['P']
    H = fp_data['h']
    
    G = -(Y - P)
    n = X.shape[1]
    grad = {}
    grad['W'] = [None] * 2
    grad['b'] = [None] * 2

    # outpute layer
    grad['W'][1] = np.dot(G, H.T) / n + 2*lam*network['W'][1]
    grad['b'][1] = np.sum(G, axis=1, keepdims=True) / n

    # hidden layer
    G = network['W'][1].T @ G
    G = G * (H > 0)

    grad['W'][0] = np.dot(G, X.T) / n + 2*lam*network['W'][0]
    grad['b'][0] = np.sum(G, axis=1, keepdims=True) / n

    return grad

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

def cyclic_learning(t, eta_min, eta_max, n_s):
    cycle_pos = t % (2 * n_s)

    if cycle_pos <= n_s:
        return eta_min + (cycle_pos/n_s) * (eta_max - eta_min)
    else:
        return eta_max - ((cycle_pos - n_s) / n_s) * (eta_max - eta_min)

def mini_batch_GD(X, Y, y, X_val, y_val, GD_params, init_net, lam, rng):
    trained_net = copy.deepcopy(init_net)
    n = X.shape[1]
    n_batch = GD_params['n_batch']
    n_epochs = GD_params['n_epochs']
    eta_min = GD_params['eta_min']
    eta_max = GD_params['eta_max']
    n_s = GD_params['n_s']

    train_costs = []
    val_costs = []
    train_losses = []
    val_losses =[]
    val_accs = []
    etas = []
    step_vec = []


    steps_per_cycle = 2 * n_s
    measure_steps = steps_per_cycle // 9

    t = 0

    for epoch in range(n_epochs):
        perm = rng.permutation(n)
        X_shuffled = X[:, perm]
        Y_shuffled = Y[:, perm]

        for j in range(n // n_batch):
            j_start = j * n_batch
            j_end = (j+1)*n_batch
            Xbatch = X_shuffled[:, j_start:j_end]
            Ybatch = Y_shuffled[:, j_start:j_end]

            eta = cyclic_learning(t, eta_min, eta_max, n_s)
            etas.append(eta)

            fp_data = apply_network(Xbatch, trained_net)
            grads = backward_pass(Xbatch, Ybatch, fp_data, trained_net, lam)


            for i in range(len(trained_net['W'])):
                trained_net['W'][i] -= eta * grads['W'][i]
                trained_net['b'][i] -= eta * grads['b'][i]
            t += 1
        
            if t % measure_steps == 0:
                fp_data_train = apply_network(X, trained_net)

                train_loss = compute_loss(fp_data_train['P'], y)
                train_losses.append(train_loss)
                train_cost = compute_cost(fp_data_train['P'], y, trained_net, lam)
                train_costs.append(train_cost)



                fp_data_val = apply_network(X_val, trained_net)

                val_loss = compute_loss(fp_data_val['P'], y_val)
                val_losses.append(val_loss)

                val_cost = compute_cost(fp_data_val['P'], y_val, trained_net, lam)
                val_costs.append(val_cost)

                val_acc = compute_accuracy(fp_data_val['P'], y_val)
                val_accs.append(val_acc)

                step_vec.append(t)

                # print(f"step {t}, training cost: {train_cost:.6f}")

    return trained_net, train_costs, val_costs, train_losses, val_losses, val_accs, etas, step_vec

def load_all_training_data():
    X_train_list, Y_train_list, y_train_list = [], [], []

    for i in range(1, 5):
        X, Y, y = load_batch(f'data_batch_{i}')
        X_train_list.append(X)
        Y_train_list.append(Y)
        y_train_list.append(y)

    X5, Y5, y5 = load_batch('data_batch_5')

    X_train_list.append(X5[:, :1000])
    Y_train_list.append(Y5[:, :1000])
    y_train_list.append(y5[:1000])

    X_val = X5[:, 1000:]
    Y_val = Y5[:, 1000:]
    y_val = y5[1000:]

    X_train = np.concatenate(X_train_list, axis=1)
    Y_train = np.concatenate(Y_train_list, axis=1)
    y_train = np.concatenate(y_train_list, axis=0)

    return X_train, Y_train, y_train, X_val, Y_val, y_val

        

def main_test_gradients():
    X_train, Y_train, y_train = load_batch('data_batch_1')
    X_val, Y_val, y_val = load_batch('data_batch_2') 
    X_test, Y_test, y_test = load_batch('test_batch')

    X_mean, X_std = compute_stats(X_train)
    X_train = normalize(X_train, X_mean, X_std)
    X_val = normalize(X_val, X_mean, X_std)
    X_test = normalize(X_test, X_mean, X_std)
    d = X_train.shape[0]
    K = Y_train.shape[0]

    d_small = 5
    n_small = 3
    m = 6
    lam = 0

    small_net, rng = init_parameters(2,d_small, m, K)

    X_small = X_train[0:d_small, 0:n_small]
    Y_small = Y_train[:, 0:n_small]
    fp_data = apply_network(X_small, small_net)
    my_grads = backward_pass(X_small, Y_small, fp_data, small_net, lam)
    torch_grads = ComputeGradsWithTorch(X_small, y_train[0:n_small], small_net)

    compare_grads(my_grads, torch_grads)


def main_small_check():
    X_train, Y_train, y_train = load_batch('data_batch_1')
    X_val, Y_val, y_val = load_batch('data_batch_2') 
    X_test, Y_test, y_test = load_batch('test_batch')

    X_mean, X_std = compute_stats(X_train)
    X_train = normalize(X_train, X_mean, X_std)
    X_val = normalize(X_val, X_mean, X_std)
    X_test = normalize(X_test, X_mean, X_std)
    d = X_train.shape[0]
    K = Y_train.shape[0]
    L = 2 
    m = 50
    init_net, rng = init_parameters(L,d,m,K)

    GD_params = {
    'n_batch': 25,
    'eta': 0.005,
    'n_epochs': 200
    }
    lam = 0

    X_small = X_train[:, 0:100]
    Y_small = Y_train[:, 0:100]
    y_small = y_train[0:100]

    print(X_small.shape)
    X_small_val = X_val[:, 0:100]
    Y_small_val = Y_val[:, 0:100]
    y_small_val = y_val[0:100]

    trained_net, train_costs, val_costs, train_losses, val_losses = mini_batch_GD(X_small, Y_small, y_small, X_small_val, y_small_val, GD_params, init_net, lam, rng)
    epochs = np.arange(1, GD_params['n_epochs']+1)

    
    plt.plot(epochs, train_costs, label='Training cost')
    plt.plot(epochs, val_costs, label='Validation cost')

    plt.xlabel('Epoch')
    plt.ylabel('Cost')
    plt.title('Training vs Validation Cost')
    plt.legend()
    plt.savefig('cost_plot.png')   

    plt.show()

    plt.figure()
    plt.plot(epochs, train_losses, label='Training loss')
    plt.plot(epochs, val_losses, label='Validation loss')

    plt.xlabel('Epoch')
    plt.ylabel('loss')
    plt.title('Training vs Validation Loss')
    plt.legend()
    plt.savefig('loss_plot.png')   

    plt.show()

def main():
    #X_train, Y_train, y_train = load_batch('data_batch_1')
    #X_val, Y_val, y_val = load_batch('data_batch_2') 

    X_train, Y_train, y_train, X_val, Y_val, y_val = load_all_training_data()
    X_test, Y_test, y_test = load_batch('test_batch')

    X_mean, X_std = compute_stats(X_train)
    X_train = normalize(X_train, X_mean, X_std)
    X_val = normalize(X_val, X_mean, X_std)
    X_test = normalize(X_test, X_mean, X_std)
    d = X_train.shape[0]
    K = Y_train.shape[0]
    L = 2 
    m = 50
    n = X_train.shape[1]
    init_net, rng = init_parameters(L,d,m,K)

    GD_params = {
    'n_batch': 100,
    'n_epochs': 24,
    'eta_min' : 1e-5,
    'eta_max' : 1e-1,
    'n_s' : 4 * np.floor(n / 100)
    }
    lam = 7.25856e-4
    print(lam)
    l_min = -3.5
    l_max = -2.5


    # Search----------------------------------------------------------------------------------------------------------------------------------------------------
    """n_samples = 20
    l = l_min + (l_max - l_min) * rng.random(1)
    lams = 10**l
    print(lams)

    results = []
    for lam in lams:
        trained_net, train_costs, val_costs, train_losses, val_losses, val_accs, etas, step_vec = mini_batch_GD(X_train, Y_train, y_train, X_val, y_val, GD_params, init_net, lam, rng)
        best_acc = max(val_accs)
        results.append({'lambda': lam,
                        'accuracy' : best_acc})
    
        print(f"lambda = {lam:.5e}, best val acc = {best_acc:.4f}")

    results = sorted(results, key=lambda x: x['accuracy'], reverse=True)

    for r in results[:5]:
        print(f"lambda = {r['lambda']:.5e}, val acc = {r['accuracy']:.4f}")"""

    #------------------------------------------------------------------------------------------------------------------------------------------------------------


    trained_net, train_costs, val_costs, train_losses, val_losses, val_accs, etas, step_vec = mini_batch_GD(X_train, Y_train, y_train, X_val, y_val, GD_params, init_net, lam, rng)
    fp_data_test = apply_network(X_test, trained_net) 
    score = compute_accuracy(fp_data_test['P'], y_test)
    

    print(score)
    epochs = np.arange(1, GD_params['n_epochs']+1)


    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # --- Cost plot ---
    axs[0].plot(step_vec, train_costs, label='Training cost')
    axs[0].plot(step_vec, val_costs, label='Validation cost')
    axs[0].set_xlabel('Update Step')
    axs[0].set_ylabel('Cost')
    axs[0].set_title('Training vs Validation Cost')
    axs[0].legend()

    # --- Loss plot ---
    axs[1].plot(step_vec, train_losses, label='Training loss')
    axs[1].plot(step_vec, val_losses, label='Validation loss')
    axs[1].set_xlabel('Update Step')
    axs[1].set_ylabel('Loss')
    axs[1].set_title('Training vs Validation Loss')
    axs[1].legend()

    plt.tight_layout()
    plt.savefig('cost_loss_subplot.png')
    plt.show()

    steps = np.arange(len(etas))

    plt.figure()
    plt.plot(steps, etas, label = 'eta')
    plt.xlabel('Epoch')
    plt.ylabel('eta')
    plt.savefig('eta_plot.png')   

    plt.show()




  
    
main()