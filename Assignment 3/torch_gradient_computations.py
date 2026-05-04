import torch
import numpy as np


def ComputeGradsWithTorch(MX, y, network_params):
    
    MXt = torch.tensor(MX)

    L = len(network_params['W'])

    # will be computing the gradient w.r.t. these parameters    
    W = [None] * L
    b = [None] * L    
    for i in range(len(network_params['W'])):
        W[i] = torch.tensor(network_params['W'][i], requires_grad=True)
        b[i] = torch.tensor(network_params['b'][i], requires_grad=True)    

    Fs_flat = torch.tensor(network_params['Fs_flat'], requires_grad=True)    

    ## give informative names to these torch classes        
    apply_relu = torch.nn.ReLU()
    apply_softmax = torch.nn.Softmax(dim=0)

    #### BEGIN your code ###########################
    
    # Apply the scoring function corresponding to equations (1-3) in assignment description 
    # If X is d x n then the final scores torch array should have size 10 x n 

    n_p = MX.shape[0]
    n = MX.shape[2]
    nf = network_params['Fs_flat'].shape[1]

    conv_outputs = torch.einsum('ijn,jl->iln', MXt, Fs_flat)

    conv_flat = torch.reshape(conv_outputs, (n_p * nf, n))
    conv_flat = apply_relu(conv_flat)


    s1 = W[0] @ conv_flat + b[0]
    x1 = apply_relu(s1)
    scores = W[1] @ x1 + b[1]


    #### END of your code ###########################            

    # apply SoftMax to each column of scores     
    P = apply_softmax(scores)
    
    # compute the loss
    loss = torch.mean(-torch.log(P[y, np.arange(n)]))
    
    # compute the backward pass relative to the loss and the named parameters 
    loss.backward()

    # extract the computed gradients and make them numpy arrays 
    grads = {}
    grads['W'] = [None] * L
    grads['b'] = [None] * L
    for i in range(L):
        grads['W'][i] = W[i].grad.numpy()
        grads['b'][i] = b[i].grad.numpy()

    grads['Fs_flat'] = Fs_flat.grad.numpy()

    return grads
