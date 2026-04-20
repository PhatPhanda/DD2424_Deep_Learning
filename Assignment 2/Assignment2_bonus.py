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
    

def apply_network(X, network, dropout_rate = 0, train=True):

    fp_data = {}
    s1 = np.dot(network['W'][0], X) + network['b'][0]

    # RELU
    h = np.maximum(0, s1)

    # Dropout if training
    if train and dropout_rate > 0:
        mask = (np.random.rand(*h.shape) > dropout_rate)
        h = h * mask
        h = h / (1 - dropout_rate) 
        fp_data['mask'] = mask
    else:
        fp_data['mask'] = None


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

def backward_pass(X, Y, fp_data, network, lam, dropout_rate = 0):
    P = fp_data['P']
    H = fp_data['h']
    mask = fp_data['mask']
    
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

    if mask is not None:
        G = G * mask
        G = G / (1 -  dropout_rate)

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

def mini_batch_GD(X, Y, y, X_val, y_val, GD_params, init_net, lam, rng, dropout_rate = 0):
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

            if GD_params.get('use_augmentation', False):
                Xbatch = augment_batch(
                    Xbatch,
                    rng,
                    mirror_prob=GD_params.get('mirror_prob', 0.5),
                    max_shift=GD_params.get('max_shift', 3)
                )

            eta = cyclic_learning(t, eta_min, eta_max, n_s)
            etas.append(eta)

            fp_data = apply_network(Xbatch, trained_net, dropout_rate, train=True)
            grads = backward_pass(Xbatch, Ybatch, fp_data, trained_net, lam, dropout_rate)


            for i in range(len(trained_net['W'])):
                trained_net['W'][i] -= eta * grads['W'][i]
                trained_net['b'][i] -= eta * grads['b'][i]
            t += 1
        
            if t % measure_steps == 0:
                fp_data_train = apply_network(X, trained_net, 0, train=False)

                train_loss = compute_loss(fp_data_train['P'], y)
                train_losses.append(train_loss)
                train_cost = compute_cost(fp_data_train['P'], y, trained_net, lam)
                train_costs.append(train_cost)



                fp_data_val = apply_network(X_val, trained_net, 0, train=False)

                val_loss = compute_loss(fp_data_val['P'], y_val)
                val_losses.append(val_loss)

                val_cost = compute_cost(fp_data_val['P'], y_val, trained_net, lam)
                val_costs.append(val_cost)

                val_acc = compute_accuracy(fp_data_val['P'], y_val)
                val_accs.append(val_acc)

                step_vec.append(t)

                # print(f"step {t}, training cost: {train_cost:.6f}")

    return trained_net, train_costs, val_costs, train_losses, val_losses, val_accs, etas, step_vec

def mini_batch_GD_adam(X, Y, y, X_val, y_val, GD_params, init_net, lam, rng, dropout_rate=0):
    trained_net = copy.deepcopy(init_net)
    n = X.shape[1]
    n_batch = GD_params['n_batch']
    n_epochs = GD_params['n_epochs']

    # Adam hyperparameters (explicit ONLY)
    eta = GD_params['eta']
    beta1 = GD_params['beta1']
    beta2 = GD_params['beta2']
    eps = GD_params['eps']

    train_costs = []
    val_costs = []
    train_losses = []
    val_losses = []
    val_accs = []
    step_vec = []

    # Adam state
    mW = [np.zeros_like(W) for W in trained_net['W']]
    vW = [np.zeros_like(W) for W in trained_net['W']]
    mb = [np.zeros_like(b) for b in trained_net['b']]
    vb = [np.zeros_like(b) for b in trained_net['b']]

    t = 0
    measure_steps = max(1, (n // n_batch) // 2)

    for epoch in range(n_epochs):
        perm = rng.permutation(n)
        X_shuffled = X[:, perm]
        Y_shuffled = Y[:, perm]

        for j in range(n // n_batch):
            j_start = j * n_batch
            j_end = (j + 1) * n_batch

            Xbatch = X_shuffled[:, j_start:j_end]
            Ybatch = Y_shuffled[:, j_start:j_end]

            if GD_params['use_augmentation']:
                Xbatch = augment_batch(
                    Xbatch,
                    rng,
                    mirror_prob=GD_params['mirror_prob'],
                    max_shift=GD_params['max_shift']
                )

            fp_data = apply_network(Xbatch, trained_net, dropout_rate, train=True)
            grads = backward_pass(Xbatch, Ybatch, fp_data, trained_net, lam, dropout_rate)

            t += 1

            for i in range(len(trained_net['W'])):
                # first moment
                mW[i] = beta1 * mW[i] + (1 - beta1) * grads['W'][i]
                mb[i] = beta1 * mb[i] + (1 - beta1) * grads['b'][i]

                # second moment
                vW[i] = beta2 * vW[i] + (1 - beta2) * (grads['W'][i] ** 2)
                vb[i] = beta2 * vb[i] + (1 - beta2) * (grads['b'][i] ** 2)

                # bias correction
                mW_hat = mW[i] / (1 - beta1 ** t)
                mb_hat = mb[i] / (1 - beta1 ** t)
                vW_hat = vW[i] / (1 - beta2 ** t)
                vb_hat = vb[i] / (1 - beta2 ** t)

                # update
                trained_net['W'][i] -= eta * mW_hat / (np.sqrt(vW_hat) + eps)
                trained_net['b'][i] -= eta * mb_hat / (np.sqrt(vb_hat) + eps)

            if t % measure_steps == 0:
                fp_data_train = apply_network(X, trained_net, 0, train=False)
                train_loss = compute_loss(fp_data_train['P'], y)
                train_cost = compute_cost(fp_data_train['P'], y, trained_net, lam)

                fp_data_val = apply_network(X_val, trained_net, 0, train=False)
                val_loss = compute_loss(fp_data_val['P'], y_val)
                val_cost = compute_cost(fp_data_val['P'], y_val, trained_net, lam)
                val_acc = compute_accuracy(fp_data_val['P'], y_val)

                train_losses.append(train_loss)
                train_costs.append(train_cost)
                val_losses.append(val_loss)
                val_costs.append(val_cost)
                val_accs.append(val_acc)
                step_vec.append(t)
                print(f"step {t}, training cost: {train_cost:.6f}")

    return trained_net, train_costs, val_costs, train_losses, val_losses, val_accs, step_vec

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

def train_and_evaluate(m, lam, dropout_rate = 0):
    X_train, Y_train, y_train, X_val, Y_val, y_val = load_all_training_data()
    X_test, Y_test, y_test = load_batch('test_batch')

    X_mean, X_std = compute_stats(X_train)
    X_train = normalize(X_train, X_mean, X_std)
    X_val = normalize(X_val, X_mean, X_std)
    X_test = normalize(X_test, X_mean, X_std)

    d = X_train.shape[0]
    K = Y_train.shape[0]
    L = 2
    n = X_train.shape[1]

    init_net, rng = init_parameters(L, d, m, K)

    GD_params = {
        'n_batch': 100,
        'n_epochs': 24,
        'eta': 1e-3,
        'beta1': 0.9,
        'beta2': 0.999,
        'eps': 1e-8,
        'use_augmentation': True,
        'mirror_prob': 0.4,
        'max_shift': 3
    }

    trained_net, train_costs, val_costs, train_losses, val_losses, val_accs, step_vec = mini_batch_GD_adam(
        X_train, Y_train, y_train, X_val, y_val, GD_params, init_net, lam, rng, dropout_rate
    )

    fp_data_test = apply_network(X_test, trained_net, 0, train=False)
    test_acc = compute_accuracy(fp_data_test['P'], y_test)

    best_val_acc = max(val_accs)

    return best_val_acc, test_acc, train_losses, val_losses, train_costs, val_costs

def vector_to_image(x):
    return x.reshape(3, 32, 32)

def mirror_image(img):
    # horizontal flip
    return img[:, :, ::-1]

def translate_image(img, tx, ty):

    xx = img.reshape(3072, 1)
    xx_shifted = np.zeros_like(xx)

    if tx >= 0:
        tx_pos = tx
        x_dest_offset = tx
        x_src_offset = 0
    else:
        tx_pos = -tx
        x_dest_offset = 0
        x_src_offset = -tx

    if ty >= 0:
        ty_pos = ty
        y_dest_offset = ty
        y_src_offset = 0
    else:
        ty_pos = -ty
        y_dest_offset = 0
        y_src_offset = -ty

    # Width/height of overlapping region
    width = 32 - tx_pos
    height = 32 - ty_pos

    if width <= 0 or height <= 0:
        return img.copy()

    aa = np.arange(height).reshape((height, 1))
    vv = np.tile(32 * aa, (1, width))

    bb_dest = np.arange(x_dest_offset, x_dest_offset + width).reshape((width, 1))
    bb_src  = np.arange(x_src_offset,  x_src_offset  + width).reshape((width, 1))

    ind_dest = vv.reshape((height * width, 1)) + np.tile(bb_dest, (height, 1)) + y_dest_offset * 32
    ind_src  = vv.reshape((height * width, 1)) + np.tile(bb_src,  (height, 1)) + y_src_offset  * 32

    inds_dest = np.vstack((ind_dest, 1024 + ind_dest))
    inds_dest = np.vstack((inds_dest, 2048 + ind_dest))

    inds_src = np.vstack((ind_src, 1024 + ind_src))
    inds_src = np.vstack((inds_src, 2048 + ind_src))

    xx_shifted[inds_dest] = xx[inds_src]

    return xx_shifted.reshape(3, 32, 32)

def augment_batch(X, rng, mirror_prob=0.5, max_shift=3):

    X_aug = np.zeros(X.shape)
    n = X.shape[1]

    for i in range(n):
        img = vector_to_image(X[:, i])

        # random mirror
        if rng.random() < mirror_prob:
            img = mirror_image(img)

        # random translation
        tx = rng.integers(-max_shift, max_shift + 1)
        ty = rng.integers(-max_shift, max_shift + 1)
        img = translate_image(img, tx, ty)

        X_aug[:, i] = img.reshape(3072)

    return X_aug

def show_augmented(X, rng):
    idx = rng.integers(0, X.shape[1])

    original = vector_to_image(X[:, idx]).transpose(1, 2, 0)
    augmented = vector_to_image(augment_batch(X[:, idx:idx+1], rng)[:, 0]).transpose(1, 2, 0)

    fig, axs = plt.subplots(1, 2, figsize=(6, 3))
    axs[0].imshow(original)
    axs[0].set_title("Original")
    axs[0].axis("off")

    axs[1].imshow(augmented)
    axs[1].set_title("Augmented")
    axs[1].axis("off")

    plt.tight_layout()
    plt.show()

def main():
    lam = 7.25856e-4
    dropout_rate = 0.25
    m_values = [500]

    results = []

    for m in m_values:
        #print(f"Testing m = {m}")
        best_val_acc, test_acc, train_losses, val_losses, train_costs, val_costs= train_and_evaluate(m, lam, dropout_rate)

        results.append({
            'm': m,
            'best_val_acc': best_val_acc,
            'test_acc': test_acc
        })

        print(f"m = {m}: best val acc = {best_val_acc:.4f}, test acc = {test_acc:.4f}")

    print("\nFinal results:")
    for r in results:
        print(r)

    epochs = np.arange(1, len(train_losses)+1)


    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # --- Cost plot ---
    axs[0].plot(epochs, train_costs, label='Training cost')
    axs[0].plot(epochs, val_costs, label='Validation cost')
    axs[0].set_xlabel('Update Step')
    axs[0].set_ylabel('Cost')
    axs[0].set_title('Training vs Validation Cost')
    axs[0].legend()

    # --- Loss plot ---
    axs[1].plot(epochs, train_losses, label='Training loss')
    axs[1].plot(epochs, val_losses, label='Validation loss')
    axs[1].set_xlabel('Update Step')
    axs[1].set_ylabel('Loss')
    axs[1].set_title('Training vs Validation Loss')
    axs[1].legend()

    plt.tight_layout()
    plt.savefig('cost_loss_subplot.png')
    plt.show()

    # steps = np.arange(len(etas))

    """plt.figure()
    plt.plot(steps, etas, label = 'eta')
    plt.xlabel('Epoch')
    plt.ylabel('eta')
    plt.savefig('eta_plot.png')   

    plt.show()"""




  
    
main()