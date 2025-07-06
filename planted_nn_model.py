import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader
import scipy.sparse as sparse
import cupy as cp
import time


def get_dims(layer, num_layers, input_dim, output_dim, m):
    if layer == 0:
        return input_dim, m
    elif layer == num_layers and output_dim > 1:  # this is to handle fusing the last layer weights
        return m, output_dim
    else:
        return m, m


# Build the planted model, assuming fused last layer if scalar output
def get_weights(num_layers, input_dim, output_dim, m, s, convex, seed):
    weights = []
    nlayers = num_layers
    if output_dim > 1:
        nlayers = num_layers + 1
    for layer in range(nlayers):
        shape1, shape2 = get_dims(layer, num_layers, input_dim, output_dim, m)
        w_idx = np.random.choice(a=shape1 * shape2, size=min(shape1*shape2, int(s / nlayers)), replace=False) # [s_layer,]
        w_values = np.random.normal(size=min(shape1*shape2, int(s / nlayers))).astype(np.float32)
        # print(f'planted indices for layer {layer} are {w_idx} with values {w_values}')
        w = np.zeros(shape1 * shape2)
        w[w_idx] = w_values
        w = np.reshape(w, (shape1, shape2))
        weights.append(w)
    return weights


def apply_model(weights, data, output_dim):
    # data is [batch, input_dim]
    for layeri, w in enumerate(weights):
        data = data @ w  # [batch, m]
        if layeri < len(weights) - 1 or output_dim == 1:  # handle fused weights
            data = np.maximum(0, data)  
    if output_dim == 1:
        return np.sum(data, axis=-1)
    return data


def make_sparse_matrix(v, d):
    # v is a SparseVector of length m * d; reshape to sparse matrix [m, d]
    indices = v.indices.get()
    values = v.values.get()
    # remove any negative indices, which can arise if a layer does not max out its sparsity capacity
    values = values[indices >= 0]
    indices = indices[indices >= 0]
    row_indices = indices // d
    col_indices = indices - row_indices * d
    mat = sparse.coo_array((values, (row_indices, col_indices)), shape=(v.n//d, d))
    return mat


def relu(x):
    return np.maximum(0, x) 


def layer_embedding(input, w1, mat=None):
    # w1 is a sparsevector of length m*d
    # Convert w1 into a cupy sparse matrix of shape (m, d)
    if mat is None:
        mat = make_sparse_matrix(v=w1, d=input.shape[-1]).T
    # assume input is shape (batch, d)
    # return relu(input @ w1mat.T), which has shape (batch, m)
    return relu(input @ mat)


class PlantedLoader(Dataset):
    def __init__(self, data, labels, ws=[]):
        self.data = data
        self.labels = labels
        self.ws = ws
        self.mats = []
        for w in self.ws:
            self.mats.append(make_sparse_matrix(v=w, d=data[0].shape[-1]).T)
        self.embedding_time = 0

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        data = self.data[idx]
        t0 = time.time()
        for (w, mat) in zip(self.ws, self.mats):
            data = layer_embedding(data, w, mat)
        self.embedding_time += time.time() - t0
        return data, self.labels[idx]


# m is the hidden dimension, s is the total number of nonzero weights, n is the number of training examples
# num_layers is the number of hidden layers, assuming the output is scalar so the output layer weights are fused
def get_planted_model_data(num_layers, input_dim, output_dim, m, s, n, seed, batch_size=0, ws=[], convex=False):
    np.random.seed(seed + 1)  # assume seed will be used to initialize the model to fit this, so pick a different seed here
    # Build a random sparse NN model
    # If we are using a convex model, we fix the random seed at initialization to match what we use in planted_nn_model.py so that the set of activation patterns matches
    weights = get_weights(num_layers, input_dim, output_dim, m, s, convex, seed)

    # Build a random input dataset
    np.random.seed(seed + 2)
    data = np.random.normal(size=(n, input_dim)).astype(np.float32)
    labels = apply_model(weights, data, output_dim).astype(np.float32)
    max_label_abs = np.max(np.abs(labels))
    if max_label_abs == 0:
        print(f'WARNING: all labels are zero')
    else:
        print(f'Maximum label magnitude is {max_label_abs}')

    torch.manual_seed(seed + 3)
    if batch_size == 0:
        batch_size = n
        train_loader = DataLoader(PlantedLoader(data=data, labels=labels, ws=ws), batch_size=batch_size, shuffle=False)  # Don't shuffle in the full-batch case
    else:
        train_loader = DataLoader(PlantedLoader(data=data, labels=labels, ws=ws), batch_size=batch_size, shuffle=True)
    return train_loader, weights, max_label_abs
