import pickle 
import numpy as np
from torch_gradient_computations import ComputeGradsWithTorch
import copy
import matplotlib.pyplot as plt


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
    z = np.dot(network['W'], X)
    s = z + network['b']

    P = np.exp(s) / np.sum(np.exp(s), axis=0, keepdims=True)
    return P

def apply_network_sigmoid(X, network):
    s = np.dot(network['W'], X) + network['b']
    P = 1 / (1 + np.exp(-s))
    return P

def compute_loss(P, y):
    n = P.shape[1]
    correct_probs = P[y, np.arange(n)]
    L = -np.mean(np.log(correct_probs))
    return L

def compute_loss_bce(P, Y):
    eps = 1e-10
    P = np.clip(P, eps, 1 - eps)
    K = Y.shape[0]
    l = -np.sum((1 - Y) * np.log(1 - P) + Y * np.log(P),axis=0) / K 

    return np.mean(l)

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

def backward_pass_sigmoid_bce(X, Y, P, network, lam):
    n = X.shape[1]
    K = Y.shape[0]

    G = (P - Y) / K 

    grad = {}
    grad['W'] = np.dot(G, X.T) / n + 2*lam*network['W']
    grad['b'] = np.sum(G, axis=1, keepdims=True) / n
    return grad

def compute_cost(P, y, network, lam):
    loss = compute_loss(P, y)
    reg = lam * np.sum(network['W'] ** 2)
    cost = loss + reg
    return cost

def compute_multiple_bce_cost(P, Y, network, lam):
    loss = compute_loss_bce(P, Y)
    reg = lam * np.sum(network['W'] ** 2)
    return loss + reg

    
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


def mini_batch_GD(X, Y, y, X_val, y_val, GD_params, init_net, lam, rng):
    trained_net = copy.deepcopy(init_net)
    n = X.shape[1]
    n_batch = GD_params['n_batch']
    eta = GD_params['eta']
    n_epochs = GD_params['n_epochs']

    train_costs = []
    val_costs = []
    train_losses = []
    val_losses =[]
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

        train_loss = compute_loss(P_train, y)
        train_losses.append(train_loss)
        train_cost = compute_cost(P_train, y, trained_net, lam)
        train_costs.append(train_cost)



        P_val = apply_network(X_val, trained_net)

        val_loss = compute_loss(P_val, y_val)
        val_losses.append(val_loss)

        val_cost = compute_cost(P_val, y_val, trained_net, lam)
        val_costs.append(val_cost)

        # print(f"Epoch {epoch+1}/{n_epochs}, training cost: {train_cost:.6f}")

    return trained_net, train_costs, val_costs, train_losses, val_losses

def mini_batch_GD_decay(X, Y, y, X_val, y_val, GD_params, init_net, lam, rng):
    trained_net = copy.deepcopy(init_net)
    n = X.shape[1]
    n_batch = GD_params['n_batch']
    eta = GD_params['eta']
    n_epochs = GD_params['n_epochs']
    step_size = GD_params.get('step_size', 10)

    train_costs = []
    val_costs = []
    train_losses = []
    val_losses = []

    for epoch in range(n_epochs):
        eta_epoch = eta * (0.1 ** (epoch // step_size))

        perm = rng.permutation(n)
        X_shuffled = X[:, perm]
        Y_shuffled = Y[:, perm]

        for j in range(n // n_batch):
            j_start = j * n_batch
            j_end = (j + 1) * n_batch
            Xbatch = X_shuffled[:, j_start:j_end]
            Ybatch = Y_shuffled[:, j_start:j_end]

            P_batch = apply_network(Xbatch, trained_net)
            grads = backward_pass(Xbatch, Ybatch, P_batch, trained_net, lam)

            trained_net['W'] -= eta_epoch * grads['W']
            trained_net['b'] -= eta_epoch * grads['b']

        P_train = apply_network(X, trained_net)
        train_loss = compute_loss(P_train, y)
        train_losses.append(train_loss)
        train_cost = compute_cost(P_train, y, trained_net, lam)
        train_costs.append(train_cost)

        P_val = apply_network(X_val, trained_net)
        val_loss = compute_loss(P_val, y_val)
        val_losses.append(val_loss)
        val_cost = compute_cost(P_val, y_val, trained_net, lam)
        val_costs.append(val_cost)

        print(f"Epoch {epoch+1}: eta = {eta_epoch}")

    return trained_net, train_costs, val_costs, train_losses, val_losses

def mini_batch_GD_sigmoid_bce(X, Y, y, X_val, Y_val, GD_params, init_net, lam, rng):
    trained_net = copy.deepcopy(init_net)
    n = X.shape[1]
    n_batch = GD_params['n_batch']
    eta = GD_params['eta']
    n_epochs = GD_params['n_epochs']

    train_costs = []
    val_costs = []
    train_losses = []
    val_losses = []

    for epoch in range(n_epochs):
        perm = rng.permutation(n)
        X_shuffled = X[:, perm]
        Y_shuffled = Y[:, perm]

        for j in range(n // n_batch):
            j_start = j * n_batch
            j_end = (j + 1) * n_batch

            Xbatch = X_shuffled[:, j_start:j_end]
            Ybatch = Y_shuffled[:, j_start:j_end]

            P_batch = apply_network_sigmoid(Xbatch, trained_net)
            grads = backward_pass_sigmoid_bce(Xbatch, Ybatch, P_batch, trained_net, lam)

            trained_net['W'] -= eta * grads['W']
            trained_net['b'] -= eta * grads['b']

        P_train = apply_network_sigmoid(X, trained_net)
        train_loss = compute_loss_bce(P_train, Y)
        train_cost = compute_multiple_bce_cost(P_train, Y, trained_net, lam)

        P_val = apply_network_sigmoid(X_val, trained_net)
        val_loss = compute_loss_bce(P_val, Y_val)
        val_cost = compute_multiple_bce_cost(P_val, Y_val, trained_net, lam)

        train_losses.append(train_loss)
        train_costs.append(train_cost)
        val_losses.append(val_loss)
        val_costs.append(val_cost)

        #print(f"Epoch {epoch+1}/{n_epochs}, train loss={train_loss:.4f}, val loss={val_loss:.4f}")

    return trained_net, train_costs, val_costs, train_losses, val_losses

def load_all_training_data():
    X_train_list, Y_train_list, y_train_list = [], [], []

    for i in range(1, 5):
        X, Y, y = load_batch(f'data_batch_{i}')
        X_train_list.append(X)
        Y_train_list.append(Y)
        y_train_list.append(y)

    X5, Y5, y5 = load_batch('data_batch_5')

    X_train_list.append(X5[:, :9000])
    Y_train_list.append(Y5[:, :9000])
    y_train_list.append(y5[:9000])

    X_val = X5[:, 9000:]
    Y_val = Y5[:, 9000:]
    y_val = y5[9000:]

    X_train = np.concatenate(X_train_list, axis=1)
    Y_train = np.concatenate(Y_train_list, axis=1)
    y_train = np.concatenate(y_train_list, axis=0)

    return X_train, Y_train, y_train, X_val, Y_val, y_val

def evaluate_hyperparams(X_train, Y_train, y_train, X_val, y_val, K, d, lam_values, eta_values, batch_values, n_epochs=40):
    results = []
    best_score = -1
    best_params = None
    best_net = None

    for lam in lam_values:
        for eta in eta_values:
            for n_batch in batch_values:
                init_net, rng = init_parameters(K, d)

                GD_params = {
                    'n_batch': n_batch,
                    'eta': eta,
                    'n_epochs': n_epochs
                }

                trained_net, train_costs, val_costs, train_losses, val_losses = mini_batch_GD(
                    X_train, Y_train, y_train, X_val, y_val,
                    GD_params, init_net, lam, rng
                )

                P_val = apply_network(X_val, trained_net)
                val_acc = compute_accuracy(P_val, y_val)

                results.append({
                    'lam': lam,
                    'eta': eta,
                    'n_batch': n_batch,
                    'val_acc': val_acc
                })

                print(f"lam={lam}, eta={eta}, n_batch={n_batch}, val_acc={val_acc:.4f}")

                if val_acc > best_score:
                    best_score = val_acc
                    best_params = {
                        'lam': lam,
                        'eta': eta,
                        'n_batch': n_batch
                    }
                    best_net = trained_net

    return best_params, best_score, best_net, results

def get_hist_data(P, y):
    gt_probs = P[y, np.arange(P.shape[1])]
    y_pred = np.argmax(P, axis=0)

    correct = gt_probs[y_pred == y]
    incorrect = gt_probs[y_pred != y]

    return correct, incorrect

def main():
    # X_train, Y_train, y_train, X_val, Y_val, y_val = load_all_training_data()
    X_train, Y_train, y_train = load_batch('data_batch_1')
    X_val, Y_val, y_val = load_batch('data_batch_2') 

    X_test, Y_test, y_test = load_batch('test_batch')

    X_mean, X_std = compute_stats(X_train)
    X_train = normalize(X_train, X_mean, X_std)
    X_val = normalize(X_val, X_mean, X_std)
    X_test = normalize(X_test, X_mean, X_std)
    d = X_train.shape[0]
    K = Y_train.shape[0]



    lam_values = [0, 0.01, 0.1, 0.2]
    eta_values = [0.0001, 0.0005, 0.001, 0.005]
    batch_values = [50, 100, 200]


    # Grid search
    """best_params, best_score, best_net, results = evaluate_hyperparams(X_train, Y_train, y_train, X_test, y_test, K, d, lam_values, eta_values, batch_values,n_epochs=40)

    print("Best params:", best_params)
    print("Best validation accuracy:", best_score)

    GD_params = {
        'n_batch': best_params['n_batch'],
        'eta': best_params['eta'],
        'n_epochs': 40
    }    """

    init_net, rng = init_parameters(K, d)
    GD_params = {
    'n_batch': 100,
    'eta': 0.001,
    'n_epochs': 40,
    'step_size' : 35
    }
    lam = 0.1    





    trained_net, train_costs, val_costs, train_losses, val_losses = mini_batch_GD(X_train, Y_train, y_train, X_val, y_val, GD_params, init_net, lam, rng
    )

    epochs = np.arange(1, GD_params['n_epochs']+1)
    

    fig, axs = plt.subplots(1, 2)

    axs[0].plot(epochs, train_costs, label='Training cost')
    axs[0].plot(epochs, val_costs, label='Validation cost')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Cost')
    axs[0].set_title('Cost')
    axs[0].legend()

    axs[1].plot(epochs, train_losses, label='Training loss')
    axs[1].plot(epochs, val_losses, label='Validation loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Loss')
    axs[1].set_title('Loss')
    axs[1].legend()

    plt.tight_layout()
    plt.savefig('cost_loss_combined.png')



    P_test = apply_network(X_test, trained_net) 
    score = compute_accuracy(P_test, y_test)
    print(score)
    
  
    Ws = trained_net['W'].transpose().reshape((32, 32, 3, 10), order='F')
    W_im = np.transpose(Ws, (1, 0, 2, 3))
    fig, axs = plt.subplots(2, 5) 
    for i in range(10):
        w_im = W_im[:, :, :, i]
        w_im_norm = (w_im - np.min(w_im)) / (np.max(w_im) - np.min(w_im))
        
        axs[i // 5, i % 5].imshow(w_im_norm)
        axs[i // 5, i % 5].axis('off')
        axs[i // 5, i % 5].set_title(f"Class {i}")

    plt.savefig('W_matrix.png')
    plt.tight_layout()
    #plt.show()

def main_bce():
    X_train, Y_train, y_train = load_batch('data_batch_1')
    X_val, Y_val, y_val = load_batch('data_batch_2')
    X_test, Y_test, y_test = load_batch('test_batch')

    X_mean, X_std = compute_stats(X_train)
    X_train = normalize(X_train, X_mean, X_std)
    X_val = normalize(X_val, X_mean, X_std)
    X_test = normalize(X_test, X_mean, X_std)

    d = X_train.shape[0]
    K = Y_train.shape[0]

    init_net, rng = init_parameters(K, d)

    GD_params = {
        'n_batch': 100,
        'eta': 0.01,
        'n_epochs': 40
    }

    lam = 0.1

    trained_net, train_costs, val_costs, train_losses, val_losses = mini_batch_GD_sigmoid_bce(X_train, Y_train, y_train, X_val, Y_val, GD_params, init_net, lam, rng)
    trained_net_soft, train_costs_soft, val_costs_soft, train_losses_soft, val_losses_soft = mini_batch_GD(X_train, Y_train, y_train, X_val, y_val, GD_params, init_net, lam, rng)
    epochs = np.arange(1, GD_params['n_epochs'] + 1)

    fig, axs = plt.subplots(1, 2)
    axs[0].plot(epochs, train_losses_soft, label='Training loss')
    axs[0].plot(epochs, val_losses_soft, label='Validation loss')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')
    axs[0].set_title('Softmax + CE loss')
    axs[0].legend()

    axs[1].plot(epochs, train_losses, label='Training loss')
    axs[1].plot(epochs, val_losses, label='Validation loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Loss')
    axs[1].set_title('Sigmoid + Multiple BCE Loss')
    axs[1].legend()

    plt.tight_layout()
    plt.savefig('sigmoid_bce_cost_loss.png')

    P_test = apply_network_sigmoid(X_test, trained_net)
    P_test_soft = apply_network(X_test, trained_net_soft)
    test_acc = compute_accuracy(P_test, y_test)
    test_acc_soft = compute_accuracy(P_test_soft, y_test)

    print("Test accuracy:", test_acc)
    print("Test accuracy soft:", test_acc_soft)


    correct_s, incorrect_s = get_hist_data(P_test_soft, y_test)
    correct_b, incorrect_b = get_hist_data(P_test, y_test)



    fig, axs = plt.subplots(1, 2, figsize = (12,5))

    # Softmax
    axs[0].hist(correct_s, bins=20, alpha=0.7, label='Correct')
    axs[0].hist(incorrect_s, bins=20, alpha=0.7, label='Incorrect')
    axs[0].set_title('Softmax + CE')
    axs[0].set_xlabel('Highest ouput probability')
    axs[0].set_ylabel('Count')
    axs[0].legend()

    # Sigmoid
    axs[1].hist(correct_b, bins=20, alpha=0.7, label='Correct')
    axs[1].hist(incorrect_b, bins=20, alpha=0.7, label='Incorrect')
    axs[1].set_title('Sigmoid + BCE')
    axs[1].set_xlabel('Highest ouput probability')
    axs[1].set_ylabel('Count')
    axs[1].legend()

    plt.tight_layout()
    plt.savefig('histogram_comparison.png')

main_bce()


