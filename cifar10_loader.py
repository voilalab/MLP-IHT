import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, Dataset
import numpy as np
import scipy.sparse as sparse

classes_to_keep = ['airplane', 'automobile']
class_indices = [0, 1]  # CIFAR-10 class indices for airplane and automobile

# Custom transform to flatten the image and add optional bias
class ToVector:
    def __init__(self, bias):
        self.bias = bias

    def __call__(self, image):
        # Convert the image to a tensor
        tensor_image = transforms.ToTensor()(image)
        # Flatten the tensor to a 1D vector
        tensor_image = tensor_image.view(-1)
        if self.bias:
            tensor_image = torch.cat((tensor_image, torch.tensor([1.0])))
        return tensor_image.view(-1)
    

def load_CIFAR(bias):
    transform = transforms.Compose([
        ToVector(bias)
    ])

    # Load the CIFAR-10 dataset
    cifar10_train = datasets.CIFAR10(root='/data/datasets', train=True, download=True, transform=transform)
    cifar10_test = datasets.CIFAR10(root='/data/datasets', train=False, download=True, transform=transform)

    # Filter dataset to only include the desired classes
    indices_train = [i for i, (_, label) in enumerate(cifar10_train) if label in class_indices]
    indices_test = [i for i, (_, label) in enumerate(cifar10_test) if label in class_indices]
    train_dataset = Subset(cifar10_train, indices_train)
    test_dataset = Subset(cifar10_test, indices_test)
    return train_dataset, test_dataset


# Create a mapping of original labels to new labels (0 and 1)
def new_label(label):
    return class_indices.index(label)


def make_sparse_matrix(v, d):
    # v is a SparseVector of length m * d; reshape to sparse matrix [m, d]
    indices = v.indices.get()
    row_indices = indices // d
    col_indices = indices - row_indices * d
    mat = sparse.coo_matrix((v.values.get(), (row_indices, col_indices)), shape=(v.n//d, d))
    return mat


def relu(x):
    return x * (x > 0)


def layer_embedding(input, w1):
    # w1 is a sparsevector of length m*d
    # Convert w1 into a cupy sparse matrix of shape (m, d)
    mat = make_sparse_matrix(v=w1, d=input.shape[-1])
    # assume input is shape (batch, d)
    # return relu(input @ w1mat.T), which has shape (batch, m)
    return relu(np.array(input) @ mat.T)


# Create a new dataset class to provide modified labels
class BinaryCIFAR10(torch.utils.data.Dataset):
    def __init__(self, dataset, ws=[], mean=[0,0,0], std=[1,1,1]):
        self.dataset = dataset
        self.ws = ws
        # apply normalization
        imgs = self.dataset.dataset.data[self.dataset.indices]
        self.dataset.dataset.data[self.dataset.indices] = (imgs - mean) / std

    def __getitem__(self, index):
        img, label = self.dataset[index]
        for w in self.ws:
            img = layer_embedding(img, w)
        return img, new_label(label)

    def __len__(self):
        return len(self.dataset)


def get_normalization(train_dataset):
    imgs = train_dataset.dataset.data[train_dataset.indices]  # [10000, 32, 32, 3]
    return np.mean(imgs, axis=(0,1,2)), np.std(imgs, axis=(0,1,2))


def get_binary_cifar10_dataloaders(bias, batch_size, ws=[], seed=0):
    torch.manual_seed(seed)
    train_dataset, test_dataset = load_CIFAR(bias)
    mean, std = get_normalization(train_dataset)
    train_dataset = BinaryCIFAR10(train_dataset, ws=ws, mean=mean, std=std)
    test_dataset = BinaryCIFAR10(test_dataset, ws=ws, mean=mean, std=std)
    if batch_size == 0:
        batch_size = len(train_dataset)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)  # Don't shuffle in the full-batch case
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def fourfeat_embedding(x, B):
    x_proj = (2.*np.pi*x) @ B.T
    return np.concatenate([np.sin(x_proj), np.cos(x_proj)], axis=-1).astype(np.float32)


class CIFAR10INRLoader(Dataset):
    def __init__(self, image, d=100, mapping_sigma=5, ws=[], seed=0):
        self.image = image.numpy().reshape((3,32,32)).transpose((1, 2, 0))  # Reshape the image to be H, W, C
        self.d = d
        np.random.seed(seed)
        # Use separate embedding sigma for space vs color channel
        self.B = np.zeros(shape=(self.d//2, 3), dtype=np.float32)
        self.B[:,0:2] = np.random.normal(size=(self.d//2, 2)) * mapping_sigma  # space embedding
        self.B[:,2] = np.random.normal(size=(self.d//2,)) * 0.05  # color embedding
        self.ws = ws
        
    def __len__(self):
        return self.image.shape[0] * self.image.shape[1] * self.image.shape[2]  # total number of pixels * channels

    def __getitem__(self, idx):
        # convert idx to a tuple for (x_index, y_index, channel_index)
        channel_index = idx // (self.image.shape[0] * self.image.shape[1])
        x_index = (idx - channel_index * self.image.shape[0] * self.image.shape[1]) // self.image.shape[1]
        y_index = idx - x_index * self.image.shape[1] - channel_index * self.image.shape[0] * self.image.shape[1]
        # First apply Fourier feature embedding to the raw coordinates
        embedding = fourfeat_embedding(x=np.array([x_index, y_index, channel_index]), B=self.B)
        # If there are any trained layer weights for initial layers, apply them as embeddings
        for w in self.ws:
            embedding = layer_embedding(embedding, w)
        value = self.image[x_index, y_index, channel_index].astype(np.float32)  # these are between 0 and 1
        return embedding, value