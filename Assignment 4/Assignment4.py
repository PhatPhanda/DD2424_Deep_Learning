import numpy as np


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

    return K, char_to_ind, ind_to_char

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
        p[:, t:t+1] = np.exp(o) / np.sum(np.exp(o), axis=0, keepdims=True)


        loss += -(y_t.T @ np.log(p[:, t:t+1]))[0,0]
        h_prev = h

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

    seq_length = X.shape[1]
    dh_next = np.zeros_like(h)


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
        

        g_t = -(y_t - p_t).T

        grads['V'] += g_t.T @ h_t.T
        grads['c'] += g_t

        dh = g_t @ V + dh_next

        da = dh @ np.diag(1 - np.tanh(a_t)**2)

        grads['W'] += da.T @ h[:, t-1:t].T
        grads['U'] += da.T @ x_t.T
        grads['b'] += da

        dh_next = da @ RNN['W']




def main():
    K, char_to_ind, ind_to_char = read_file()
    m = 100
    eta = .001
    seq_length = 25

    RNN, rng = init_parameters(K,m)

    h0 = np.zeros((m, 1))

    x0 = np.zeros((K, 1))
    x0[char_to_ind["."], 0] = 1

    Y = synthesize_text(RNN, h0, x0, K, 200, rng)

    generated_indices = np.argmax(Y, axis=0)
    generated_text = ''.join(ind_to_char[i] for i in generated_indices)

    print(generated_text)



main()

    