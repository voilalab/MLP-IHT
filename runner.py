import argparse
from enum import Enum
import cupy as cp
import torch
import time
import os
from torch.utils.data import DataLoader


class Task(Enum):
    MNIST = 'mnist'
    BINARY_MNIST = 'binary_mnist'
    BINARY_CIFAR10 = 'binary_cifar10'
    MLP_PLANTED = 'mlp_planted'
    INR_MNIST = 'inr_mnist'
    INR_CIFAR10 = 'inr_cifar10'
    GAUSSIAN_SPARSE = 'gaussian_sparse'


def task_type(value):
    try:
        return Task[value.upper()]  # Converts string to enum
    except KeyError:
        raise argparse.ArgumentTypeError(f"Invalid task: {value}. Must be one of {[task.name for task in Task]}.")


parser = argparse.ArgumentParser()
parser.add_argument('--method', type=str, default='aiht',
                    help='What method to train. Options are aiht or pruning')
parser.add_argument('--prune_iters', type=float, default=0.1,
                    help='Controls the type of pruning to use to achieve the desired model sparsity, if --method is pruning. '
                         '0 means post-training magnitude pruning. '
                         'A value in (0, 1) means IMP pruning that fraction of weights each iteration. '
                         'A value >=1 means IMP pruning a constant fraction each time, for that total number of iterations, '
                         'with the amount pruned each time chosen so that the final sparsity is as desired.') # Note that 0.2 is somewhat standard, but 0.1 is usually better (but slower) so we use 0.1
parser.add_argument('--task', type=task_type, 
                    help=f'What type of sparse fitting task to run. Options are {[task.name for task in Task]}')
parser.add_argument('--convex', action='store_true',
                    help='Use a convex neural network (default is nonconvex NN). Only relevant if the task involves a NN.')
parser.add_argument('--bias', action='store_true',
                    help='Add a bias term to the NN. Only relevant if the task involves a NN.')
parser.add_argument('--batch_size', type=int, default=0, 
                    help='Batch size to use for stochastic gradients. Default=0 denotes full-batch gradients.')
parser.add_argument('--num_epochs', type=int, default=15, 
                    help='Number of epochs for training.')
parser.add_argument('--num_layers', type=int, default=1,
                    help='Number of hidden layers, trained sequentially if using AIHT.')
parser.add_argument('--output_dim', type=int, default=10,
                    help='Dimension of the output layer, if the task is mlp_planted.')
parser.add_argument('--mlist', nargs='+', type=int, 
                    help='How many hidden neurons (layer width) to use for tasks that involve a NN, \
                        or the length of the unknown signal for Gaussian sensing. \
                        You may provide one or more values of m to test.')
parser.add_argument('--slist', nargs='+', type=int, 
                    help='How many nonzero parameters to allow. You may provide one or more values of s to test.')
parser.add_argument('--seedlist', nargs='+', type=int,
                    help='Direct control over random seed, used for initializing dataloaders and models. You may provide one or more random seeds to test.')
parser.add_argument('--nmeasurements', type=int, default=50000,
                    help='How many measurements to use, if the task is gaussian_sparse or mlp_planted.')
parser.add_argument('--input_dim', type=int, default=100,
                    help='Dimension of the input vectors, if the task is mlp_planted.')
parser.add_argument('--save_results', action='store_true',
                    help='Save a table with the results of this experiment for each value of m, s, and seed.')
parser.add_argument('--gpu', type=int, default=0,
                    help='Which GPU to run on.')
parser.add_argument('--expname', type=str, default='',
                    help='Optional unique string to identify this experiment. Will be appended to the log file name if results are saved.')
args = parser.parse_args()

if args.seedlist is None or len(args.seedlist) == 0:
    args.seedlist = [0]

cp.cuda.Device(args.gpu).use()
torch.cuda.set_device(f'cuda:{args.gpu}')


from mnist_loader import get_mnist_dataloaders, load_MNIST, MNISTINRLoader
from cifar10_loader import get_binary_cifar10_dataloaders, load_CIFAR, CIFAR10INRLoader
from planted_nn_model import get_planted_model_data
from sparse_AIHT import binaryExperiment, INRExperiment, GaussianExperiment
from baselines import test_model_s



t0 = time.time()
test_accs = cp.zeros((len(args.mlist), len(args.slist), len(args.seedlist)))



if args.task is Task.MNIST:
    print(f"""doing multiclass MNIST classification for {args.num_epochs} epochs with batch size {args.batch_size},
          m={args.mlist}, and s={args.slist}. Convex is {args.convex} and bias is {args.bias}.""")
    # Fit the model for each pair of m and s values
    for mi in range(len(args.mlist)):
        for seedi in range(len(args.seedlist)):
            if args.method == 'pruning':
                # Fetch the MNIST dataset--fresh each time to preserve random seed
                train_loader, test_loader = get_mnist_dataloaders(bias=args.bias, batch_size=args.batch_size, seed=args.seedlist[seedi], binary=False)
                test_accs[mi, :, seedi] = test_model_s(m=args.mlist[mi], s_s=args.slist, num_iters=args.prune_iters, train_loader=train_loader, test_loader=test_loader, input_dim=28*28, output_dim=10, num_layers=args.num_layers, bias=False, num_epochs=args.num_epochs, seed=args.seedlist[seedi], crossentropy=True)  # Here we say bias is false because it would be already wrapped into the dataloader by appending a 1
            else:
                for si in range(len(args.slist)):
                    layer_weights = []
                    for layer in range(args.num_layers):
                        # Fetch the MNIST dataset--fresh each time because things are apparently sensitive to random seed
                        train_loader, test_loader = get_mnist_dataloaders(bias=args.bias, batch_size=args.batch_size, ws=layer_weights, seed=args.seedlist[seedi], binary=False)
                        acc, x_hat = binaryExperiment(m=args.mlist[mi], s=int(args.slist[si] / args.num_layers), train_loader=train_loader, test_loader=test_loader, convex=args.convex, num_epochs=args.num_epochs, seed=args.seedlist[seedi], crossentropy=True)
                        print(f'total time in countsketch was {x_hat.countsketch.total_time} seconds')
                        # note: this choice of s is strict (will result in fewer nonzeros) because each layer will lose the parameters allocated to its temporary output layer
                        d = 784
                        if args.bias:
                            d = 785
                        layer_weights.append(x_hat.extract_block(start_idx=0, stop_idx=args.mlist[mi]*d))
                    test_accs[mi, si, seedi] = acc

elif args.task is Task.BINARY_MNIST:
    print(f"""doing binary MNIST classification for {args.num_epochs} epochs with batch size {args.batch_size},
          m={args.mlist}, and s={args.slist}. Convex is {args.convex} and bias is {args.bias}.""")
    # Fit the model for each pair of m and s values
    for mi in range(len(args.mlist)):
        for seedi in range(len(args.seedlist)):
            if args.method == 'pruning':
                # Fetch the binary MNIST dataset--fresh each time to preserve random seed
                train_loader, test_loader = get_mnist_dataloaders(bias=args.bias, batch_size=args.batch_size, seed=args.seedlist[seedi], binary=True)
                test_accs[mi, :, seedi] = test_model_s(m=args.mlist[mi], s_s=args.slist, num_iters=args.prune_iters, train_loader=train_loader, test_loader=test_loader, input_dim=28*28, output_dim=1, num_layers=args.num_layers, bias=False, num_epochs=args.num_epochs, seed=args.seedlist[seedi])  # Here we say bias is false because it would be already wrapped into the dataloader by appending a 1
            else:
                for si in range(len(args.slist)):
                    layer_weights = []
                    for layer in range(args.num_layers):
                        # Fetch the binary MNIST dataset--fresh each time to preserve random seed
                        train_loader, test_loader = get_mnist_dataloaders(bias=args.bias, batch_size=args.batch_size, ws=layer_weights, seed=args.seedlist[seedi], binary=True)
                        acc, x_hat = binaryExperiment(m=args.mlist[mi], s=int(args.slist[si] / args.num_layers), train_loader=train_loader, test_loader=test_loader, convex=args.convex, num_epochs=args.num_epochs, seed=args.seedlist[seedi])
                        layer_weights.append(x_hat)
                    test_accs[mi, si, seedi] = acc

elif args.task is Task.BINARY_CIFAR10:
    print(f"""doing binary CIFAR10 classification for {args.num_epochs} epochs with batch size {args.batch_size},
          m={args.mlist}, and s={args.slist}. Convex is {args.convex} and bias is {args.bias}.""")
    # Fit the model for each pair of m and s values
    for mi in range(len(args.mlist)):
        for seedi in range(len(args.seedlist)):
            if args.method == 'pruning': 
                # Fetch the binary CIFAR10 dataset--fresh each time to preserve random seed
                train_loader, test_loader = get_binary_cifar10_dataloaders(bias=args.bias, batch_size=args.batch_size, seed=args.seedlist[seedi])
                test_accs[mi, :, seedi] = test_model_s(m=args.mlist[mi], s_s=args.slist, num_iters=args.prune_iters, train_loader=train_loader, test_loader=test_loader, input_dim=32*32*3, output_dim=1, num_layers=args.num_layers, bias=False, num_epochs=args.num_epochs, seed=args.seedlist[seedi])  # Here we say bias is false because it would be already wrapped into the dataloader by appending a 1
            else:
                for si in range(len(args.slist)):
                    layer_weights = []
                    for layer in range(args.num_layers):
                        # Fetch the binary CIFAR10 dataset--fresh each time to preserve random seed
                        train_loader, test_loader = get_binary_cifar10_dataloaders(bias=args.bias, batch_size=args.batch_size, ws=layer_weights, seed=args.seedlist[seedi])
                        acc, x_hat = binaryExperiment(m=args.mlist[mi], s=int(args.slist[si] / args.num_layers), train_loader=train_loader, test_loader=test_loader, convex=args.convex, num_epochs=args.num_epochs, seed=args.seedlist[seedi])
                        layer_weights.append(x_hat)
                    test_accs[mi, si, seedi] = acc

elif args.task is Task.MLP_PLANTED:
    print(f"""fitting a planted model for {args.num_epochs} epochs with batch size {args.batch_size}. The model has {args.num_layers} layers,
           m={args.mlist}, s={args.slist}, and the dataset has {args.nmeasurements} examples with input dim {args.input_dim} and output dim {args.output_dim}.""")

    # note: this is testing data overfitting (not exact model fitting) for a planted MLP; train and test sets are the same

    for mi in range(len(args.mlist)):
        for si in range(len(args.slist)):
            if args.num_layers > 1:
                if args.mlist[mi]**2 < int(args.slist[si] / args.num_layers):
                    print(f'skipping m={args.mlist[mi]}, s={args.slist[si]} with {args.num_layers} layers, because the hidden layers would be more than dense')
                    continue
            for seedi in range(len(args.seedlist)):
                if args.method == 'pruning':
                    train_loader, weights, max_label_abs = get_planted_model_data(num_layers=args.num_layers, input_dim=args.input_dim, output_dim=args.output_dim, m=args.mlist[mi], s=args.slist[si], n=args.nmeasurements, seed=args.seedlist[seedi], batch_size=args.batch_size)
                    if max_label_abs == 0:
                        print(f'skipping seed {seedi} because max label norm is 0 in the planted model dataset')
                        continue
                    test_accs[mi, si, seedi] = test_model_s(m=args.mlist[mi], s_s=[args.slist[si]], num_iters=args.prune_iters, train_loader=train_loader, test_loader=train_loader, input_dim=args.input_dim, output_dim=args.output_dim, num_layers=args.num_layers, bias=False, num_epochs=args.num_epochs, seed=args.seedlist[seedi], inr_mode=True, max_label=max_label_abs)  # Here we say bias is false because it would be already wrapped into the dataloader by appending a 1
                else:
                    layer_weights = []
                    for layer in range(args.num_layers):
                        train_loader, weights, max_label_abs = get_planted_model_data(num_layers=args.num_layers, input_dim=args.input_dim, output_dim=args.output_dim, m=args.mlist[mi], s=args.slist[si], n=args.nmeasurements, seed=args.seedlist[seedi], batch_size=args.batch_size, ws=layer_weights)
                        if max_label_abs == 0:
                            print(f'skipping seed {seedi} because max label norm is 0 in the planted model dataset')
                            acc = 0
                            break
                        acc, x_hat = INRExperiment(m=args.mlist[mi], s=int(args.slist[si] / args.num_layers), train_loader=train_loader, convex=args.convex, num_epochs=args.num_epochs, seed=args.seedlist[seedi], max_label=max_label_abs)
                        # handle vector output case, which involves throwing out the intermediate output layers
                        if args.output_dim == 1:
                            layer_weights.append(x_hat)
                        else:
                            d = args.input_dim
                            if layer > 0:
                                d = args.mlist[mi]
                            layer_weights.append(x_hat.extract_block(start_idx=0, stop_idx=args.mlist[mi] * d))
                        idx = cp.nonzero(cp.array(weights[layer].T.flatten()))[0]
                    test_accs[mi, si, seedi] = acc               

elif args.task is Task.INR_MNIST:
    print(f"""fitting an MNIST image with an INR for {args.num_epochs} epochs with batch size {args.batch_size},
          m={args.mlist}, and s={args.slist}. Convex is {args.convex} and for this task we do not use bias.""")

    train, train_labels, test, test_labels = load_MNIST(bias=False)
    batch_size = args.batch_size
    if args.batch_size == 0:
        batch_size = test[1000].shape[0]  # image is already flattened

    for mi in range(len(args.mlist)):
        for seedi in range(len(args.seedlist)):
            if args.method == 'pruning': 
                torch.manual_seed(args.seedlist[seedi])
                baseloader = MNISTINRLoader(image=test[1000], d=1500, mapping_sigma=2, seed=args.seedlist[seedi])
                train_loader = DataLoader(baseloader, batch_size=batch_size, shuffle=True)
                test_loader = DataLoader(baseloader, batch_size=batch_size, shuffle=False)
                test_accs[mi, :, seedi] = test_model_s(m=args.mlist[mi], s_s=args.slist, num_iters=args.prune_iters, train_loader=train_loader, test_loader=test_loader, input_dim=2*len(baseloader.B), output_dim=1, num_layers=args.num_layers, bias=False, num_epochs=args.num_epochs, inr_mode=True, seed=args.seedlist[seedi])
            else:
                for si in range(len(args.slist)):
                    layer_weights = []
                    for layer in range(args.num_layers):
                        torch.manual_seed(args.seedlist[seedi])
                        train_loader = DataLoader(MNISTINRLoader(image=test[1000], d=1500, mapping_sigma=2, ws=layer_weights, seed=args.seedlist[seedi]), batch_size=batch_size, shuffle=True)
                        psnr, x_hat = INRExperiment(m=args.mlist[mi], s=int(args.slist[si] / args.num_layers), train_loader=train_loader, convex=args.convex, num_epochs=args.num_epochs, seed=args.seedlist[seedi])
                        layer_weights.append(x_hat)
                    test_accs[mi, si, seedi] = psnr

elif args.task is Task.INR_CIFAR10:
    print(f"""fitting a CIFAR10 image with an INR for {args.num_epochs} epochs with batch size {args.batch_size},
          m={args.mlist}, and s={args.slist}. Convex is {args.convex} and for this task we do not use bias.""")

    train_dataset, test_dataset = load_CIFAR(bias=False)
    batch_size = args.batch_size
    if args.batch_size == 0:
        batch_size = len(test_dataset[1000][0].flatten())
    for mi in range(len(args.mlist)):
        for seedi in range(len(args.seedlist)):
            if args.method == 'pruning': 
                torch.manual_seed(args.seedlist[seedi])
                baseloader = CIFAR10INRLoader(image=test_dataset[1000][0], d=1500, mapping_sigma=5, seed=args.seedlist[seedi])
                train_loader = DataLoader(baseloader, batch_size=batch_size, shuffle=True)
                test_loader = DataLoader(baseloader, batch_size=batch_size, shuffle=False)
                test_accs[mi, :, seedi] = test_model_s(m=args.mlist[mi], s_s=args.slist, num_iters=args.prune_iters, train_loader=train_loader, test_loader=test_loader, input_dim=2*len(baseloader.B), output_dim=1, num_layers=args.num_layers, bias=False, num_epochs=args.num_epochs, inr_mode=True, seed=args.seedlist[seedi])
            else:
                for si in range(len(args.slist)):
                    layer_weights = []
                    for layer in range(args.num_layers):
                        torch.manual_seed(args.seedlist[seedi])
                        train_loader = DataLoader(CIFAR10INRLoader(image=test_dataset[1000][0], d=1500, mapping_sigma=5, ws=layer_weights, seed=args.seedlist[seedi]), batch_size=batch_size, shuffle=True)
                        psnr, x_hat = INRExperiment(m=args.mlist[mi], s=int(args.slist[si] / args.num_layers), train_loader=train_loader, convex=args.convex, num_epochs=args.num_epochs, seed=args.seedlist[seedi])
                        layer_weights.append(x_hat)
                    test_accs[mi, si, seedi] = psnr

elif args.task is Task.GAUSSIAN_SPARSE:
    print(f"""fitting a generic sparse vector with {args.nmeasurements} Gaussian measurements for {args.num_epochs} full-batch steps.
          Signal has {args.mlist} entries, and s={args.slist}. This task is always convex and without bias.""")
    assert args.method == 'aiht', "This task can only be done with AIHT"
    
    # In this notation m is number of measurements and n is the total size of x
    for mi in range(len(args.mlist)):
        for si in range(len(args.slist)):
            for seedi in range(len(args.seedlist)):
                torch.manual_seed(args.seedlist[seedi])
                l2_error = GaussianExperiment(n=args.mlist[mi], s=args.slist[si], m=args.nmeasurements, num_steps=args.num_epochs, seed=args.seedlist[seedi])
                test_accs[mi, si, seedi] = l2_error

if args.save_results:
    foldername = os.path.join(args.task.name, args.method)
    os.makedirs(foldername, exist_ok=True)
    print(f'saving results in folder {foldername}')
    expname = f'hidden{args.num_layers}_m{args.mlist}_s{args.slist}_seed{args.seedlist}_convex{args.convex}_bs{args.batch_size}_bias{args.bias}_{args.num_epochs}epochs_{args.expname}'
    cp.save(os.path.join(foldername, expname), test_accs)
print(f'total time was {time.time() - t0}')

