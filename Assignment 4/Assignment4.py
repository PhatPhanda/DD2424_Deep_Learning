import numpy as np
from torch_gradient_computations_column_wise import ComputeGradsWithTorch
import matplotlib.pyplot as plt

def read_file():
    book_fname = 'Assignment 4/goblet_book.txt'
    fid = open(book_fname, "r")
    book_data = fid.read()
    fid.close()
    unique_chars = list(set(book_data))
    K = len(unique_chars)
    char_to_ind = {}
    ind_to_char = {}

    for i in range(K):
        char_to_ind[unique_chars[i]] = i
        ind_to_char[i] = unique_chars[i]

    return book_data, K, char_to_ind, ind_to_char

def init_parameters(K, m):
    rng = np.random.default_rng(42)
    # get the BitGenerator used by default_rng
    BitGen = type(rng.bit_generator)
    # use the state from a fresh bit generator
    seed = 42

    RNN = {'b': np.zeros((m,1)),
           'c': np.zeros((K,1)),
           'U': (1/np.sqrt(2*K))*rng.standard_normal(size = (m, K)),
           'W': (1/np.sqrt(2*m))*rng.standard_normal(size = (m, m)),
           'V': (1/np.sqrt(m))*rng.standard_normal(size = (K, m))      
           }


    return RNN, rng

def chars_to_one_hot(chars, char_to_ind, K):
    X = np.zeros((K, len(chars)))

    for t, ch in enumerate(chars):
        X[char_to_ind[ch], t] = 1

    return X

def chars_to_indices(chars, char_to_ind):
    return np.array([char_to_ind[ch] for ch in chars])

def one_hot_to_text(Y, ind_to_char):
    idx = np.argmax(Y, axis=0)
    chars = [ind_to_char[i] for i in idx]
    return ''.join(chars)


def synthesize_text(RNN, h0, x0, K, n, rng):
    h = h0
    x = x0

    Y = np.zeros((K,n))
    for t in range(n):
        a = RNN['W'] @ h + RNN['U'] @ x + RNN['b']
        h = np.tanh(a)
        o = RNN['V'] @ h + RNN['c']
        p = np.exp(o) / np.sum(np.exp(o), axis=0, keepdims=True)

        cp = np.cumsum(p, axis=0)
        a = rng.uniform(size=1)
        ii = np.argmax(cp - a > 0)

        x = np.zeros((K, 1))
        x[ii, 0] = 1

        Y[:, t:t+1] = x

    return Y

def forward_pass(RNN, X, Y, h0):
    seq_length = X.shape[1]

    m = RNN['W'].shape[0]
    K = RNN["V"].shape[0]

    a = np.zeros((m, seq_length))
    h = np.zeros((m, seq_length))
    o = np.zeros((K, seq_length))
    p = np.zeros((K, seq_length))

    h_prev = h0
    loss = 0

    for t in range(seq_length):
        x_t = X[:, t:t+1]
        y_t = Y[:, t:t+1]

        a[:, t:t+1] = RNN["W"] @ h_prev + RNN["U"] @ x_t + RNN["b"]
        h[:, t:t+1] = np.tanh(a[:, t:t+1])
        o[:, t:t+1] = RNN["V"] @ h[:, t:t+1] + RNN["c"]
        p[:, t:t+1] = np.exp(o[:, t:t+1]) / np.sum(np.exp(o[:, t:t+1]), axis=0, keepdims=True)


        loss += -(y_t.T @ np.log(p[:, t:t+1]))[0,0]
        h_prev = h[:, t:t+1]

    loss = loss / seq_length

    fp_data = {
        "X": X,
        "Y": Y,
        "a": a,
        "h": h,
        "o": o,
        "p": p,
        "h0": h0
    }

    return loss, fp_data


def backward_pass(RNN, fp_data):
    b = RNN['b']
    c = RNN['c']
    W = RNN['W']
    V = RNN['V']
    U = RNN['U']

    X = fp_data['X']
    Y = fp_data['Y']
    a = fp_data['a']
    h = fp_data['h']
    o = fp_data['o']
    p = fp_data['p']
    h0 = fp_data['h0']

    seq_length = X.shape[1]
    dh_next = np.zeros_like(h[:, 0:1])


    # init grads
    grads = {}

    grads["U"] = np.zeros_like(RNN["U"])
    grads["W"] = np.zeros_like(RNN["W"])
    grads["V"] = np.zeros_like(RNN["V"])
    grads["b"] = np.zeros_like(RNN["b"])
    grads["c"] = np.zeros_like(RNN["c"])

    
    for t in reversed(range(seq_length)):
        x_t = X[:, t:t+1]
        y_t = Y[:, t:t+1]
        p_t = p[:, t:t+1]
        h_t = h[:, t:t+1]
        a_t = a[:, t:t+1]
        
        if t == 0:
            h_prev = h0

        else:
            h_prev = h[:, t-1:t]

        g_t = -(y_t - p_t)

        grads['V'] += g_t @ h_t.T
        grads['c'] += g_t

        dh = V.T @ g_t + dh_next

        da = dh * (1 - h_t**2)

        grads['W'] += da @ h_prev.T
        grads['U'] += da @ x_t.T
        grads['b'] += da

        dh_next = W.T @ da
        

    for key in grads:
        grads[key] /= seq_length

    return grads

def compare_grads(my_grads, torch_grads):
    for key in ['W', 'U', 'b', 'V', 'c']:
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

def init_adam(RNN):
    m_adam = {}
    v_adam = {}

    for key in RNN:
        m_adam[key] = np.zeros_like(RNN[key])
        v_adam[key] = np.zeros_like(RNN[key])

    return m_adam, v_adam


def adam_update(RNN, grads, m_adam, v_adam, t, eta=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
    for key in RNN:
        m_adam[key] = beta1 * m_adam[key] + (1 - beta1) * grads[key]
        v_adam[key] = beta2 * v_adam[key] + (1 - beta2) * (grads[key] ** 2)

        m_hat = m_adam[key] / (1 - beta1 ** t)
        v_hat = v_adam[key] / (1 - beta2 ** t)

        RNN[key] -= eta * m_hat / (np.sqrt(v_hat) + eps)

    return RNN, m_adam, v_adam

def train_rnn(num_updates, m, seq_length, eta, print_every, synth_every, synth_length):
    book_data, K, char_to_ind, ind_to_char = read_file()

    RNN, rng = init_parameters(K, m)

    m_adam, v_adam = init_adam(RNN)

    hprev = np.zeros((m, 1))
    e = 0

    smooth_loss = None
    smooth_losses = []

    best_loss = np.inf
    best_RNN = None

    for update_step in range(1, num_updates + 1):

        if e + seq_length + 1 >= len(book_data):
            e = 0
            hprev = np.zeros((m, 1))

        X_chars = book_data[e:e + seq_length]
        Y_chars = book_data[e + 1:e + seq_length + 1]

        X = chars_to_one_hot(X_chars, char_to_ind, K)
        Y = chars_to_one_hot(Y_chars, char_to_ind, K)

        loss, fp_data = forward_pass(RNN, X, Y, hprev)
        grads = backward_pass(RNN, fp_data)

        RNN, m_adam, v_adam = adam_update(
            RNN, grads, m_adam, v_adam, update_step, eta=eta
        )

        hprev = fp_data["h"][:, -1:]

        if smooth_loss is None:
            smooth_loss = loss
        else:
            smooth_loss = 0.999 * smooth_loss + 0.001 * loss

        smooth_losses.append(smooth_loss)

        if smooth_loss < best_loss:
            best_loss = smooth_loss
            best_RNN = {key: RNN[key].copy() for key in RNN}

        if update_step % print_every == 0:
            print(f"iter = {update_step}, smooth_loss = {smooth_loss}")

        if update_step == 1 or update_step % synth_every == 0:
            print("\n--------------------------------")
            print(f"Sample at iteration {update_step}")
            print("--------------------------------")

            x0 = X[:, 0:1]
            Y_synth = synthesize_text(RNN, hprev, x0, K, synth_length, rng)
            txt = one_hot_to_text(Y_synth, ind_to_char)

            print(txt)
            print("--------------------------------\n")

        e += seq_length

    return RNN, best_RNN, smooth_losses, char_to_ind, ind_to_char

def main_test_grads():
    book_data, K, char_to_ind, ind_to_char = read_file()

    m = 10
    seq_length = 25

    RNN, rng = init_parameters(K, m)

    h0 = np.zeros((m, 1))

    X_chars = book_data[0:seq_length]
    Y_chars = book_data[1:seq_length+1]

    X = chars_to_one_hot(X_chars, char_to_ind, K)
    Y = chars_to_one_hot(Y_chars, char_to_ind, K)

    y = chars_to_indices(Y_chars, char_to_ind)

    loss, fp_data = forward_pass(RNN, X, Y, h0)

    my_grads = backward_pass(RNN, fp_data)
    torch_grads = ComputeGradsWithTorch(X, y, h0, RNN)

    print("Loss:", loss)
    compare_grads(my_grads, torch_grads)

def main():
    num_updates = 200000
    m = 100
    seq_length = 25
    eta = 0.001
    print_every = 50000
    synth_every = 50000
    synth_length= 200
    RNN, best_RNN, smooth_losses, char_to_ind, ind_to_char = train_rnn(num_updates, m, seq_length, eta, print_every, synth_every, synth_length)

    plt.figure(figsize=(10, 5))
    plt.plot(smooth_losses)
    plt.xlabel("Update step")
    plt.ylabel("Smooth loss")
    plt.title("Smooth loss during RNN training")
    plt.grid(True)
    plt.show()

    book_data, K, char_to_ind, ind_to_char = read_file()

    h0 = np.zeros((best_RNN["W"].shape[0], 1))

    start_char = book_data[0]
    x0 = np.zeros((K, 1))
    x0[char_to_ind[start_char], 0] = 1

    _, rng = init_parameters(K, best_RNN["W"].shape[0])

    Y_synth = synthesize_text(best_RNN, h0, x0, K, 1000, rng)
    generated_text = one_hot_to_text(Y_synth, ind_to_char)

    print(generated_text)

if __name__ == '__main__':
    main()
