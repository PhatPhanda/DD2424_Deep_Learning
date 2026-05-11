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
           'c': np.zeros((K,1))
           }
    RNN['U'] = (1/np.sqrt(2*K))*rng.standard_normal(size = (m, K))
    RNN['W'] = (1/np.sqrt(2*m))*rng.standard_normal(size = (m, m))
    RNN['V'] = (1/np.sqrt(m))*rng.standard_normal(size = (K, m))

    return RNN, rng

def main():
    K, char_to_ind, ind_to_char = read_file()
    m = 100
    eta = .001
    seq_length = 25

    RNN, rng = init_parameters(K,m)

    print(RNN['b'])

main()

    