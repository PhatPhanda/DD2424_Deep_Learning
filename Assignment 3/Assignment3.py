import pickle 
import numpy as np
import copy
import matplotlib.pyplot as plt
import torch
from torch_gradient_computations import ComputeGradsWithTorch
import time



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

    net_params['b_conv'] = np.zeros((nf, 1))

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
    b_conv = network['b_conv']

    n_p = MX.shape[0]
    n = MX.shape[2]
    nf = Fs_flat.shape[1]


    #First Layer
    conv_outputs = np.einsum('ijn, jl ->iln', MX, Fs_flat, optimize=True) + b_conv.reshape(1, nf, 1)

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

def backward_pass(Y, fp_data, MX, network, lam):
    Fs_flat = network['Fs_flat']
    W1 = network['W'][0]
    W2 = network['W'][1]

    conv_flat = fp_data['conv_flat']
    x1 = fp_data['x1']
    P = fp_data['P']

    n = Y.shape[1]
    n_p = MX.shape[0]
    nf = Fs_flat.shape[1]  

    grads = {} 
    grads['W'] = [None] * 2
    grads['b'] = [None] * 2
    grads['b_conv'] = None

    G = -(Y - P)

    # outpute layer
    grads['W'][1] = np.dot(G, x1.T) / n
    grads['W'][1] += 2*lam*W2

    grads['b'][1] = np.sum(G, axis=1, keepdims=True) / n

    # hidden layer
    G = W2.T @ G
    G = G * (x1 > 0)    

    grads['W'][0] = (G @ conv_flat.T) / n
    grads['W'][0] += 2*lam*W1

    grads['b'][0] = np.sum(G, axis=1, keepdims=True) / n
    G_batch = W1.T @ G                

    G_batch = G_batch * (conv_flat > 0)

    # Undo flattening
    GG = G_batch.reshape((n_p, nf, n), order='C')

    # Gradient wrt Fs_flat
    MXt = np.transpose(MX, (1, 0, 2))
    grads['Fs_flat'] = np.einsum('ijn, jln -> il', MXt, GG, optimize=True) / n
    grads['Fs_flat'] += 2 * lam * Fs_flat

    grads['b_conv'] = np.sum(GG, axis=(0, 2)).reshape(nf, 1) / n

    return grads    


def compute_cost(P, y, network, lam):
    loss = compute_loss(P, y)
    reg = lam * (np.sum(network['Fs_flat']**2) + np.sum(network['W'][0]**2) + np.sum(network['W'][1]**2))
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

    for key in ['Fs_flat', 'b_conv']:
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


def load_all_training_data():
    X_train_list, Y_train_list, y_train_list = [], [], []

    val_n = 1000

    for i in range(1, 5):
        X, Y, y = load_batch(f'data_batch_{i}')
        X_train_list.append(X)
        Y_train_list.append(Y)
        y_train_list.append(y)

    X5, Y5, y5 = load_batch('data_batch_5')

    X_train_list.append(X5[:, :val_n])
    Y_train_list.append(Y5[:, :val_n])
    y_train_list.append(y5[:val_n])

    X_val = X5[:, val_n:]
    Y_val = Y5[:, val_n:]
    y_val = y5[val_n:]

    X_train = np.concatenate(X_train_list, axis=1)
    Y_train = np.concatenate(Y_train_list, axis=1)
    y_train = np.concatenate(y_train_list, axis=0)

    return X_train, Y_train, y_train, X_val, Y_val, y_val

def cyclic_learning(t, eta_min, eta_max, n_s):
    cycle_pos = t % (2 * n_s)

    if cycle_pos <= n_s:
        return eta_min + (cycle_pos / n_s) * (eta_max - eta_min)
    else:
        return eta_max - ((cycle_pos - n_s) / n_s) * (eta_max - eta_min)


def mini_batch_GD(MX, Y, y, MX_val, y_val, GD_params, init_net, lam, rng):
    trained_net = copy.deepcopy(init_net)

    n = MX.shape[2]
    n_batch = GD_params['n_batch']
    eta_min = GD_params['eta_min']
    eta_max = GD_params['eta_max']
    n_s = GD_params['n_s']
    n_cycles = GD_params['n_cycles']

    train_costs = []
    val_costs = []
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    etas = []
    step_vec = []

    total_steps = 2 * n_s * n_cycles
    measure_steps = max(1, n_s // 2)

    t = 0

    while t < total_steps:
        perm = rng.permutation(n)

        for j in range(n // n_batch):
            if t >= total_steps:
                break

            batch_idx = perm[j * n_batch : (j + 1) * n_batch]

            MXbatch = MX[:, :, batch_idx]
            Ybatch = Y[:, batch_idx]

            eta = cyclic_learning(t, eta_min, eta_max, n_s)
            etas.append(eta)

            fp_data = forward_pass(MXbatch, trained_net)
            grads = backward_pass(Ybatch, fp_data, MXbatch, trained_net, lam)

            for i in range(len(trained_net['W'])):
                trained_net['W'][i] -= eta * grads['W'][i]
                trained_net['b'][i] -= eta * grads['b'][i]

            trained_net['Fs_flat'] -= eta * grads['Fs_flat']
            trained_net['b_conv'] -= eta * grads['b_conv']

            t += 1

            if t % measure_steps == 0:
                fp_data_train = forward_pass(MX, trained_net)

                train_loss = compute_loss(fp_data_train['P'], y)
                train_cost = compute_cost(fp_data_train['P'], y, trained_net, lam)
                train_acc = compute_accuracy(fp_data_train['P'], y)

                train_losses.append(train_loss)
                train_costs.append(train_cost)
                train_accs.append(train_acc)

                fp_data_val = forward_pass(MX_val, trained_net)

                val_loss = compute_loss(fp_data_val['P'], y_val)
                val_cost = compute_cost(fp_data_val['P'], y_val, trained_net, lam)
                val_acc = compute_accuracy(fp_data_val['P'], y_val)

                val_losses.append(val_loss)
                val_costs.append(val_cost)
                val_accs.append(val_acc)

                step_vec.append(t)

    return (
        trained_net,
        train_costs,
        val_costs,
        train_losses,
        val_losses,
        train_accs,
        val_accs,
        etas,
        step_vec
    )

def cyclic_learning_increasing(cycle_t, eta_min, eta_max, n_s):
    if cycle_t <= n_s:
        return eta_min + (cycle_t / n_s) * (eta_max - eta_min)
    else:
        return eta_max - ((cycle_t - n_s) / n_s) * (eta_max - eta_min)

def mini_batch_GD_increasing(MX, Y, y, MX_val, y_val, GD_params, init_net, lam, rng):
    trained_net = copy.deepcopy(init_net)


    n = MX.shape[2]
    n_batch = GD_params['n_batch']
    eta_min = GD_params['eta_min']
    eta_max = GD_params['eta_max']

    curr_n_s = GD_params['n_s']
    n_cycles = GD_params['n_cycles']

    train_costs = []
    val_costs = []
    train_losses = []
    val_losses =[]
    val_accs = []
    train_accs = []
    etas = []
    step_vec = []

    t = 0
    cycle_t = 0
    cycle_count = 0

    while cycle_count < n_cycles:
        perm = rng.permutation(n)

        for j in range(n // n_batch):
            if cycle_count >= n_cycles:
                break

            batch_idx = perm[j * n_batch : (j + 1) * n_batch]

            MXbatch = MX[:, :, batch_idx]
            Ybatch = Y[:, batch_idx]

            eta = cyclic_learning_increasing(cycle_t, eta_min, eta_max, curr_n_s)
            etas.append(eta)

            fp_data = forward_pass(MXbatch, trained_net)
            grads = backward_pass(Ybatch, fp_data, MXbatch, trained_net, lam)


            for i in range(len(trained_net['W'])):
                trained_net['W'][i] -= eta * grads['W'][i]
                trained_net['b'][i] -= eta * grads['b'][i]

            trained_net['Fs_flat'] -= eta * grads['Fs_flat']
            trained_net['b_conv'] -= eta * grads['b_conv']

            t += 1
            cycle_t += 1
        
            if cycle_t % max(1, curr_n_s // 2) == 0:
                fp_data_train = forward_pass(MX, trained_net)

                train_loss = compute_loss(fp_data_train['P'], y)
                train_losses.append(train_loss)

                train_cost = compute_cost(fp_data_train['P'], y, trained_net, lam)
                train_costs.append(train_cost)

                train_acc = compute_accuracy(fp_data_train['P'], y)
                train_accs.append(train_acc)


                fp_data_val = forward_pass(MX_val, trained_net)

                val_loss = compute_loss(fp_data_val['P'], y_val)
                val_losses.append(val_loss)

                val_cost = compute_cost(fp_data_val['P'], y_val, trained_net, lam)
                val_costs.append(val_cost)

                val_acc = compute_accuracy(fp_data_val['P'], y_val)
                val_accs.append(val_acc)

                step_vec.append(t)

                # print(f"step {t}, training cost: {train_cost:.6f}")

            if cycle_t >= 2 * curr_n_s:
                cycle_count += 1
                cycle_t = 0
                curr_n_s *= 2


    return trained_net, train_costs, val_costs, train_losses, val_losses, train_accs, val_accs, etas, step_vec

def compute_accuracy(P, y):
    y_pred = np.argmax(P, axis=0)
    accuracy = np.mean(y_pred == y)

    return accuracy

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
    n_debug = 8
    lam = 0.05

    network, rng = init_parameters(f, nf, nh, K)

    X_small = X_train[:, :n_debug]
    Y_small = Y_train[:, :n_debug]
    y_small = y_train[:n_debug]

    MX = create_MX(X_small, f)

    fp_data = forward_pass(MX, network)
    my_grad = backward_pass(X_small,Y_small, fp_data, MX, network, lam)
    torch_grads = ComputeGradsWithTorch(MX, y_small, network, lam)

    compare_grads(my_grad, torch_grads)

def main_exercise3_short():
    # load in datasets
    X_train, Y_train, y_train, X_val, Y_val, y_val = load_all_training_data()
    X_test, Y_test, y_test = load_batch('test_batch')

    X_mean, X_std = compute_stats(X_train)
    X_train = normalize(X_train, X_mean, X_std)
    X_val = normalize(X_val, X_mean, X_std)
    X_test = normalize(X_test, X_mean, X_std)

    # architechtures
    architectures = [
        {'f': 4,  'nf': 10,  'nh': 50},
        {'f': 8,  'nf': 40,  'nh': 50},
    ]

    

    # cyclical learning
    GD_params = {
    'n_cycles': 3,
    'n_s': 800,
    'eta_min' : 1e-5,
    'eta_max' : 1e-1,
    'n_batch' : 100
    }
    lam = 0.003
    K = 10

    test_accs = []
    training_times = []
    labels = []

    for arch in architectures:
        f = arch['f']
        nf = arch['nf']
        nh = arch['nh']

        print(f"\nTraining architecture: f={f}, nf={nf}, nh={nh}")

        MX_train = create_MX(X_train, f)
        MX_val = create_MX(X_val, f)
        MX_test = create_MX(X_test, f)

        network, rng = init_parameters(f, nf, nh, K)

        start_time = time.time()

        trained_net, train_costs, val_costs, train_losses, val_losses, train_accs, val_accs, etas, step_vec = mini_batch_GD_increasing(
            MX_train, Y_train, y_train, MX_val, y_val, GD_params, network, lam, rng
        )

        end_time = time.time()
        training_time = end_time - start_time

        fp_data_test = forward_pass(MX_test, trained_net)
        test_acc = compute_accuracy(fp_data_test['P'], y_test)

        print(f"Training time: {training_time:.2f} seconds")
        print(f"Test accuracy: {test_acc:.4f}")

        labels.append(f"f={f}, nf={nf}")
        test_accs.append(test_acc)
        training_times.append(training_time)

    # plot
    """# --- Cost & Loss subplot ---
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    axs[0].plot(step_vec, train_costs, label='Training cost')
    axs[0].plot(step_vec, val_costs, label='Validation cost')
    axs[0].set_xlabel('Update Step')
    axs[0].set_ylabel('Cost')
    axs[0].set_title('Training vs Validation Cost')
    axs[0].legend()

    axs[1].plot(step_vec, train_losses, label='Training loss')
    axs[1].plot(step_vec, val_losses, label='Validation loss')
    axs[1].set_xlabel('Update Step')
    axs[1].set_ylabel('Loss')
    axs[1].set_title('Training vs Validation Loss')
    axs[1].legend()

    plt.tight_layout()
    plt.savefig('Assignment 3/cost_loss_subplot.png')
    plt.show()


    # --- Eta plot ---
    steps = np.arange(len(etas))
    plt.figure()
    plt.plot(steps, etas, label='eta')
    plt.xlabel('Update Step')
    plt.ylabel('Learning rate')
    plt.title('Cyclical Learning Rate')
    plt.legend()
    plt.savefig('Assignment 3/eta_plot.png')
    plt.show()


    # --- Accuracy plot ---
    plt.figure()
    plt.plot(step_vec, val_accs, label='Validation Accuracy')
    plt.plot(step_vec, train_accs, label='Validation Accuracy')

    plt.xlabel('Update Step')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy')
    plt.legend()
    plt.savefig('Assignment 3/accuracy_plot.png')
    plt.show()"""



    """fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # --- Test accuracy ---
    axs[0].bar(labels, test_accs)
    axs[0].set_xlabel('Architecture')
    axs[0].set_ylabel('Final Test Accuracy')
    axs[0].set_title('Test Accuracy')
    axs[0].set_ylim(0, 1)

    # --- Training time ---
    axs[1].bar(labels, training_times)
    axs[1].set_xlabel('Architecture')
    axs[1].set_ylabel('Training Time [seconds]')
    axs[1].set_title('Training Time')

    plt.tight_layout()
    plt.savefig('Assignment 3/architecture_comparison.png')
    plt.show()"""

def main_exercise3_increasing():
    X_train, Y_train, y_train, X_val, Y_val, y_val = load_all_training_data()
    X_test, Y_test, y_test = load_batch('test_batch')

    X_mean, X_std = compute_stats(X_train)
    X_train = normalize(X_train, X_mean, X_std)
    X_val = normalize(X_val, X_mean, X_std)
    X_test = normalize(X_test, X_mean, X_std)

    architectures = [
        {'f': 4,  'nf': 40,  'nh': 50},
        {'f': 8,  'nf': 40,  'nh': 50},
    ]

    GD_params = {
        'n_cycles': 3,
        'n_s': 800,
        'eta_min': 1e-5,
        'eta_max': 1e-1,
        'n_batch': 100
    }

    lam = 0.003
    K = 10

    curves = {}

    for arch in architectures:
        f = arch['f']
        nf = arch['nf']
        nh = arch['nh']

        label = f"f={f}, nf={nf}, nh={nh}"
        print(f"\nTraining architecture: {label}")

        MX_train = create_MX(X_train, f)
        MX_val = create_MX(X_val, f)
        MX_test = create_MX(X_test, f)

        network, rng = init_parameters(f, nf, nh, K)

        start_time = time.time()

        trained_net, train_costs, val_costs, train_losses, val_losses, train_accs, val_accs, etas, step_vec = mini_batch_GD_increasing(
            MX_train, Y_train, y_train, MX_val, y_val, GD_params, network, lam, rng
        )

        end_time = time.time()
        training_time = end_time - start_time

        fp_data_test = forward_pass(MX_test, trained_net)
        test_acc = compute_accuracy(fp_data_test['P'], y_test)

        print(f"Training time: {training_time:.2f} seconds")
        print(f"Test accuracy: {test_acc:.4f}")

        curves[label] = {
            "step_vec": step_vec,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "test_acc": test_acc,
            "training_time": training_time
        }

    # --- One subplot per architecture ---
    fig, axs = plt.subplots(1, len(curves), figsize=(12, 5), sharey=True)

    if len(curves) == 1:
        axs = [axs]

    for ax, (label, data) in zip(axs, curves.items()):
        ax.plot(data["step_vec"], data["train_losses"], label="Training loss")
        ax.plot(data["step_vec"], data["val_losses"], label="Validation loss")

        ax.set_title(label)
        ax.set_xlabel("Update Step")
        ax.set_ylabel("Loss")
        ax.legend()

    plt.tight_layout()
    plt.savefig("Assignment 3/longer_training_loss_subplots.png")
    plt.show()






main_exercise3_increasing()