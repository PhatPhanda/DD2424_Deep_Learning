import pickle 
import numpy as np
import copy
import matplotlib.pyplot as plt
import torch

debug_file = 'Assignment 3\debug_info.npz'
load_data = np.load(debug_file)
X = load_data['X']
Fs = load_data['Fs']



n = X.shape[1]
f = Fs.shape[0]
nf = Fs.shape[3]
n_p = 64

X_ims = np.transpose(X.reshape((32, 32, 3, X.shape[1]), order='F'), (1, 0, 2, 3))
MX = np.zeros((n_p, f*f*3, n))
conv_outputs_mat = np.zeros((n_p,nf,n))

conv_outputs = np.zeros((32//f, 32//f, nf, n))

for i in range(n):
    l = 0
    for x_cor in range(32//f):
        for y_cor in range(32//f):
            x_start = x_cor * f
            x_end = x_start + f

            y_start = y_cor * f
            y_end = y_start + f

            X_patch = X_ims[x_start:x_end, y_start: y_end, :, i]


            MX[l, :, i] = X_patch.reshape((1, f*f*3), order='C')

            l += 1

            for k in range(nf):
                conv_outputs[x_cor, y_cor, k, i] = np.sum(X_patch * Fs[:,:,:,k])

                
Fs_flat = Fs.reshape((f*f*3, nf), order='C')

conv_outputs_mat = np.einsum('ijn, jl ->iln', MX, Fs_flat, optimize=True)

conv_outputs_flat = conv_outputs.reshape((n_p, nf, n), order='C')

print(conv_outputs_mat.shape)
print(conv_outputs_flat.shape)
print(np.allclose(conv_outputs_flat, conv_outputs_mat, atol=1e-10))
print(np.max(np.abs(conv_outputs_flat - conv_outputs_mat)))
