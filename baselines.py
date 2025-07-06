# Baselines are modified from https://pytorch.org/tutorials/intermediate/pruning_tutorial.html
import torch
from torch import nn
import torch.nn.utils.prune as prune
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import cupy as cp
from copy import deepcopy
import time
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Modified from https://colab.research.google.com/github/bentrevett/pytorch-image-classification/blob/master/1_mlp.ipynb#scrollTo=lAqzcW9XREvu
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, num_layers, m, bias, seed):
        # num_layers is the number of hidden layers, so the total number of layers is num_layers + 1
        super().__init__()
        self.seed = seed
        torch.manual_seed(seed)  # Performance is very sensitive to random seed, so fix it for reproducibility
        self.bias = bias
        self.m = m
        self.num_layers = num_layers

        self.input_fc = nn.Linear(input_dim, m, bias=bias)
        
        self.hidden_layers = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.hidden_layers.append(nn.Linear(m, m, bias=bias))
        
        self.output_fc = nn.Linear(m, output_dim, bias=bias)

        self.num_params = input_dim * m + m * m * (self.num_layers - 1) + m * output_dim
        if self.bias:
            self.num_params += m * self.num_layers + output_dim

    def forward(self, x):
        # x = [batch size, height, width]
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)  # [batch size, height * width]
        h = F.relu(self.input_fc(x))  # [batch size, m]
        for hidden_layer in self.hidden_layers:
            h = F.relu(hidden_layer(h))  # [batch size, m]
        y_pred = self.output_fc(h)  # [batch size, output dim]
        return y_pred
    
    def parameters_to_prune(self):
        params = [(self.input_fc, 'weight')]
        if self.bias:
            params.append((self.input_fc, 'bias'))
        
        for hidden_layer in self.hidden_layers:
            params.append((hidden_layer, 'weight'))
            if self.bias:
                params.append((hidden_layer, 'bias'))
        
        params.append((self.output_fc, 'weight'))
        if self.bias:
            params.append((self.output_fc, 'bias'))
        
        return params

    def params(self):
        params = [self.input_fc.weight]
        if self.bias:
            params.append(self.input_fc.bias)
        
        for hidden_layer in self.hidden_layers:
            params.append(hidden_layer.weight)
            if self.bias:
                params.append(hidden_layer.bias)
        
        params.append(self.output_fc.weight)
        if self.bias:
            params.append(self.output_fc.bias)
        
        return params
    
    def sparsity(self):
        zeros = 0
        total = 0
        for param in self.params():
            zeros = zeros + float(torch.sum(param == 0))
            total = total + float(param.nelement())
        return total - zeros
    
    def prune_magnitude(self, s):
        # Estimate the fraction to prune so that s parameters are left nonzero
        amount = 1 - s / self.num_params
        amount = max(amount, 0)
        amount = min(amount, 1)
        # This will prune the smallest weights by magnitude
        # eg if amount is 0.2 then 80% of the weights will be intact
        prune.global_unstructured(
            self.parameters_to_prune(),
            pruning_method=prune.L1Unstructured,
            amount=amount,
        )


def initialize(seed, m, input_dim, output_dim, num_layers, bias, crossentropy):
    model = MLP(input_dim=input_dim, output_dim=output_dim, num_layers=num_layers, m=m, bias=bias, seed=seed).to(device=device)
    if crossentropy:
        loss_fn = nn.CrossEntropyLoss()
    else:
        loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # SGD would be a closer comparison to IHT, but to make a stronger baseline we use Adam
    return model, loss_fn, optimizer


def train_one_epoch(model, train_loader, loss_fn, optimizer):
    for i, data in enumerate(train_loader):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs.to(device)).squeeze()
        loss = loss_fn(outputs, labels.to(device).to(outputs.dtype))
        loss.backward()
        optimizer.step()
    return


# Iterative magnitude pruning (lottery ticket style pruning)
def IMP(model, train_loader, test_loader, loss_fn, optimizer, s_s, num_iters, num_epochs, eval_func):
    s_s = np.sort(s_s)  # make sure s values are in increasing order
    # Fraction of parameters to prune each iteration
    if num_iters < 1:
        amount_i = num_iters  # Standard IMP reduces weights by 20% each time. Pruning more slowly works better but takes more compute time.
    else:
        amount_i = 1 - (s_s[-1] / model.num_params)**(1/(num_iters - 1))  
    s_i = (1 - amount_i) * model.num_params
    initialization_weights = deepcopy(model.state_dict())  # Rewind to initialization since that seems to usually work better (but there are some settings where rewinding to epoch 1 is better)
    # train_one_epoch(model, train_loader, loss_fn, optimizer)
    # initialization_weights = deepcopy(model.state_dict())  # Save the weights after one epoch of training, for rewinding
    actual_s = model.sparsity()
    if actual_s <= s_s[0]:
        newmodel = train_model(model=model, train_loader=train_loader, test_loader=test_loader, loss_fn=loss_fn, optimizer=optimizer, s=s_i, num_epochs=num_epochs)
        acc = eval_func(newmodel, test_loader=test_loader)
        if newmodel.sparsity() <= s_s[0]:
            return newmodel, cp.ones(len(s_s)) * acc

    i = 0
    accs = cp.zeros(len(s_s))
    next_sparsity_idx = 1
    while actual_s > s_s[0]:
        i = i + 1
        # Rewind the weights to their earlier values, while keeping the pruning mask
        if i > 1:
            sd = model.state_dict()
            for item in initialization_weights.keys():
                sd[item + "_orig"] = initialization_weights[item]
            model.load_state_dict(sd)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Reset the optimizer each time 
        # Train with this amount of magnitude pruning
        model = train_model(model=model, train_loader=train_loader, test_loader=test_loader, loss_fn=loss_fn, optimizer=optimizer, s=s_i, num_epochs=num_epochs)
        acc = eval_func(model, test_loader=test_loader)
        old_actual_s = actual_s
        actual_s = model.sparsity()
        # Increase the pruning percent if the actual sparsity is not decreasing. This can happen if e.g. actual_sparsity=5 and s_i=0.1, since 0.5<1 and we cannot prune a fraction of a parameter.
        if actual_s >= old_actual_s:
            amount_i = 1.0 / actual_s  # Ensure that we prune at least one parameter
            s_i = (1 - amount_i) * model.num_params
        print(f'iter {i} of IMP has actual s={actual_s} on the way to s={s_s[0]}, with test performance {acc}')
        while actual_s <= s_s[-next_sparsity_idx]:
            accs[-next_sparsity_idx] = acc
            next_sparsity_idx = next_sparsity_idx + 1
            if next_sparsity_idx > len(s_s):
                break
    return model, accs


def eval_acc(model, test_loader):
    correct = 0
    total = 0
    for i, data in enumerate(test_loader):
        inputs, labels = data
        outputs = model(inputs.to(device)).squeeze()
        if len(outputs.shape) == 1:
            outputs[outputs >= 0.5] = 1
            outputs[outputs < 0.5] = 0
            correct = correct + torch.sum(outputs == labels.to(device))
        else:
            outputs = torch.argmax(outputs, axis=1)
            labels = torch.argmax(labels.to(device), axis=1)
            correct = correct + torch.sum(outputs == labels)
        total = total + len(labels)
    return (correct / total).item()


def eval_psnr(model, test_loader):
    total_mse = 0
    count = 0
    for i, data in enumerate(test_loader):
        inputs, labels = data
        prediction = model(inputs.to(device)).squeeze()
        total_mse += torch.mean(torch.square(prediction - labels.to(device)))
        count += 1
    psnr = -10*torch.log10(total_mse / count)
    return psnr.item()


def train_model(model, train_loader, test_loader, loss_fn, optimizer, s, num_epochs):
    loader = train_loader
    # Special case for training when there are not many batches, since the dataloader overhead dominates runtime in that setting
    if len(train_loader) <= 100:
        iterator = iter(train_loader)
        loader = []
        for _ in range(len(train_loader)):
            train, train_labels = next(iterator)
            loader.append((train, train_labels))
    for epoch in range(num_epochs):
        train_one_epoch(model, loader, loss_fn, optimizer)
    model.prune_magnitude(s=s)
    return model


def test_model_s(m, s_s, num_iters, train_loader, test_loader, input_dim, output_dim, num_layers, bias, num_epochs, inr_mode=False, seed=0, crossentropy=False, max_label=1):
    eval_func = eval_acc
    if inr_mode:
        def eval_func(model, test_loader):
            return 20*np.log10(max_label) + eval_psnr(model=model, test_loader=test_loader)
    # For magnitude thresholding without retraining
    if num_iters == 0:
        accs = cp.zeros(len(s_s))
        for i in range(len(s_s)):
            model, loss_fn, optimizer = initialize(seed=seed, m=m, input_dim=input_dim, output_dim=output_dim, num_layers=num_layers, bias=bias, crossentropy=crossentropy)  # reset for the next experiment
            model = train_model(model=model, train_loader=train_loader, test_loader=test_loader, loss_fn=loss_fn, optimizer=optimizer, s=s_s[i], num_epochs=num_epochs)
            accs[i] = eval_func(model, test_loader=test_loader)
    # For IMP
    elif num_iters > 0:
        model, loss_fn, optimizer = initialize(seed=seed, m=m, input_dim=input_dim, output_dim=output_dim, num_layers=num_layers, bias=bias, crossentropy=crossentropy)
        model, accs = IMP(model=model, train_loader=train_loader, test_loader=test_loader, loss_fn=loss_fn, optimizer=optimizer, s_s=s_s, num_iters=num_iters, num_epochs=num_epochs, eval_func=eval_func)
    else:
        return False, f'num_iters must be nonnegative'
    return accs
