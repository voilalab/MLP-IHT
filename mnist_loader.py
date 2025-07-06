import cupy as cp
from scipy.io import loadmat
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import scipy.sparse as sparse


def load_MNIST(bias, k=10):
    mnist = loadmat("/data/datasets/mnist/mnist-original.mat")
    mnist_label = cp.asarray(mnist["label"][0])  # 70k
    mnist_data = cp.asarray(mnist["data"].T)  # (70k, 784)
    if bias:
        # Append 1 to each image to create a bias term
        mnist_data = cp.concatenate((cp.asarray(mnist_data), cp.ones((len(mnist_data), 1))), axis=1)  # (70k, 785)
    # Take exactly 5000 train and 1000 test for each digit
    train = []
    train_labels = []
    test = []
    test_labels = []
    for digit in range(k):
        data = mnist_data[mnist_label==digit,:]
        train.append(data[0:5000,:])
        train_labels.append(cp.ones(5000)*digit)
        test.append(data[5000:6000,:])
        test_labels.append(cp.ones(1000)*digit)
    train = cp.concatenate(train)  # [5000*k, 784/5]
    train_labels = cp.concatenate(train_labels)  # 5000*k
    test = cp.concatenate(test)  # [1000*k, 784/5]
    test_labels = cp.concatenate(test_labels)  # 1000*k
    # Make the labels one-hot encoded
    eye = cp.eye(k)
    train_labels = eye[train_labels.astype(int)]  # [5000*k, k]
    test_labels = eye[test_labels.astype(int)]  # [1000*k, k]
    return train, train_labels, test, test_labels

    
def load_binary_MNIST(bias):
    mnist = loadmat("/data/datasets/mnist/mnist-original.mat")
    mnist_label = mnist["label"][0]  # 70k
    mnist_data = mnist["data"].T  # (70k, 784)
    if bias:
        # Append 1 to each image to create a bias term
        mnist_data = cp.concatenate((cp.array(mnist_data), cp.ones((len(mnist_data), 1))), axis=1)  # (70k, 785)
    # Filter to only keep the 0s and 1s
    zeros = mnist_data[mnist_label==0,:]  # (6903, 784/5)
    ones = mnist_data[mnist_label==1,:]  # (7877, 784/5)
    # Keep the first 5k as train and the next 1k as test
    zeros_train = zeros[0:5000,:]
    zeros_test = zeros[5000:6000,:]
    ones_train = ones[0:5000,:]
    ones_test = ones[5000:6000,:]
    train = cp.concatenate((cp.array(zeros_train), cp.array(ones_train)))  # (10k, 784/5)
    test = cp.concatenate((cp.array(zeros_test), cp.array(ones_test)))  # (2k, 784/5)
    train_labels = cp.concatenate((cp.zeros(5000), cp.ones(5000)))
    test_labels = cp.concatenate((cp.zeros(1000), cp.ones(1000)))
    return train, train_labels, test, test_labels


def make_sparse_matrix(v, d):
    # v is a SparseVector of length m * d; reshape to sparse matrix [m, d]
    indices = v.indices.get()
    keepidx = indices >= 0
    indices = indices[keepidx]
    values = v.values.get()[keepidx]
    row_indices = indices // d
    col_indices = indices - row_indices * d
    mat = sparse.coo_array((values, (row_indices, col_indices)), shape=(v.n//d, d))
    return mat


def relu(x):
    return x * (x > 0)


def layer_embedding(input, w1):
    # w1 is a sparsevector of length m*d
    # Convert w1 into a cupy sparse matrix of shape (m, d)
    mat = make_sparse_matrix(v=w1, d=input.shape[-1])
    # assume input is shape (batch, d)
    # return relu(input @ w1mat.T), which has shape (batch, m)
    return relu(input @ mat.T) 


class MNISTLoader(Dataset):
    def __init__(self, data, labels, ws=[], mean=0, std=1):
        self.data = (data - mean) / std
        self.labels = labels
        self.ws = ws

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.data[idx].get().astype(np.float32)
        for w in self.ws:
            image = layer_embedding(image, w)
        label = self.labels[idx].get().astype(np.float32)
        return image, label


def fourfeat_embedding(x, B):
    x_proj = (2.*np.pi*x) @ B.T
    return np.concatenate([np.sin(x_proj), np.cos(x_proj)], axis=-1).astype(np.float32)


class MNISTINRLoader(Dataset):
    def __init__(self, image, d=100, mapping_sigma=1, ws=[], seed=0):
        self.image = image.get().reshape((28,28))  # Reshape the image to be MNIST sized
        self.d = d
        np.random.seed(seed)
        self.B = np.random.normal(size=(self.d//2, 2)) * mapping_sigma
        self.ws = ws
        
    def __len__(self):
        return self.image.shape[0] * self.image.shape[1]

    def __getitem__(self, idx):
        # convert idx to a tuple for (x_index, y_index)
        x_index = idx // self.image.shape[1]
        y_index = idx - x_index * self.image.shape[1]
        # First apply Fourier feature embedding to the raw coordinates
        embedding = fourfeat_embedding(x=np.array([x_index, y_index]), B=self.B)
        # If there are any trained layer weights for initial layers, apply them as embeddings
        for w in self.ws:
            embedding = layer_embedding(embedding, w)
        value = self.image[x_index, y_index].astype(np.float32) / 255.  # between 0 and 1
        return embedding, value


def get_mnist_dataloaders(bias, batch_size, ws=[], seed=0, binary=True):
    torch.manual_seed(seed)
    if binary:
        train, train_labels, test, test_labels = load_binary_MNIST(bias=bias)
    else:
        train, train_labels, test, test_labels = load_MNIST(bias=bias, k=10)
    if batch_size == 0:
        batch_size = len(train)
        train_loader = DataLoader(MNISTLoader(data=train, labels=train_labels, ws=ws), batch_size=batch_size, shuffle=False) # Don't shuffle in the full-batch case
    else:
        train_loader = DataLoader(MNISTLoader(data=train, labels=train_labels, ws=ws), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(MNISTLoader(data=test, labels=test_labels, ws=ws), batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

