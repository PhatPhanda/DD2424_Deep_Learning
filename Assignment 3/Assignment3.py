import pickle 
import numpy as np
import copy
import matplotlib.pyplot as plt
import torch

debug_file = 'Assignment 3\debug_info.npz'
load_data = np.load(debug_file)
X = load_data['X']
Fs = load_data['Fs']

print(X.shape[1])