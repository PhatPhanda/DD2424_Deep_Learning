import numpy as np
from torch_gradient_computations_column_wise import ComputeGradsWithTorch

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
    rng = np.random.default_rng()
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




main_test_grads()

    