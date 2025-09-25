This code is associated with the following paper, and reproduces the experimental results presented there: 

A Recovery Guarantee for Sparse Neural Networks

https://arxiv.org/abs/2509.20323

## How to run the code

This codebase uses the following Python packages: numpy, torch, torchvision, cupy, cupyx, scipy, argparse, matplotlib, time, tqdm, copy, enum, os, and glob.

The primary file for running experiments is `runner.py`, which has many command-line flags that allow you to control which experiment is run.

### Common command-line flags

`--method` controls which algorithm to run; options are `aiht` for Accelerated Iterative Hard Thresholding and `pruning` for Iterative Magnitude Pruning (IMP, the algorithm from the Lottery Ticket Hypothesis)

`--task` controls what the MLP fitting task is, among the following options: `mnist`, `binry_mnist`, `binary_cifar10`, `mlp_planted`, `inr_mnist`, `inr_cifar10`, and `gaussian_sparse`. All of these are MLP fitting tasks, with the exception of `gaussian_sparse` which is a sanity check using IMP to fit sparse vectors with Gaussian sensing matrices.

`--mlist` controls the width of the hidden layer(s) in the MLP. For the `gaussian_sparse` task, this controls the length of the unknown signal. You may list one or more integer values; if multiple values are given then each will be tested sequentially.

`--slist` controls the maximum number of nonzero values (the sparsity level). You may list one or more integer values; if multiple values are given then each will be tested sequentially.

`--batch_size` controls the batch size for computing (stochastic) gradients. By convention we use `0` to denote full-batch gradients (this is the default); other integer values (if provided) denote the minibatch size.

`--num_epochs` controls how many epochs (full passes through the dataset) of optimization to run.

`--num_layers` controls the number of hidden layers (default=1) in the MLP. If the method is IHT, each hidden layer is trained sequentially so the training time will increase.

`--seedlist` controls the random seed used for initializing dataloaders and models, to enable reproducibility. You may list one or more integer values; if multiple values are given then each will be tested sequentially.

`--save_results` gives the option to save a numpy table with the results of this experiment, for subsequent analysis and figure generation. If this flag is not provided, results will be printed in the terminal but not saved. 

`--expname` gives the option to provide a unique string to identify this experiment. If results are saved, this string will be appended to the log file name, which automatically includes the `task`, `method`, `num_layers`, `mlist`, `slist`, random seed (from `seedlist`), `convex`, `batch_size`, `bias`, and `num_epochs`.

### Less common command-line flags

`--gpu` controls which GPU to run on, if multiple are available. The default is `0`.

`--input_dim` controls the input (data) dimension of the MLP, if the task is `mlp_planted`. For other tasks the input dimension is controlled automatically by the dataset.

`--output_dim` controls the output dimension of the last layer of the MLP, if the task is `mlp_planted`. For other tasks the output dimension is controlled automatically by the dataset.

`--nmeasurements` controls how many measurements to use (the number of examples in the training dataset), if the task is `mlp_planted` or `gaussian_sparse`.

`--prune_iters` controls how exactly iterative pruning is done for IMP (our baseline); `0` denotes post-training magnitude pruning, a value in `(0,1)` denotes IMP pruning that fraction of active weights at each iteration, a value `>= 1` denotes pruning for exactly that many iterations, with the pruning fraction at each iteration calculated to achieve the desired sparsity level after a fixed number of steps. Our default (used in all experiments) is `0.1`, which does IMP by pruning 10% of the active weights in each step, for as many steps as needed to reach the desired sparsity level.

`--convex` gives the option to use a fully convex formulation, without sequential convex updates to the sensing matrix A. This setting matches our theoretical setting except for using randomly sampled h vectors inside A, rather than enumerating all activation patterns. Our experiments with IHT all use the sequential convex formulation (i.e., omitting this flag).

`--bias` gives the option to add a bias term to the MLP. Our experiments did not use bias.

### Examples

We provide examples corresponding to each figure in the paper.

Figure 1
```
python runner.py --task mlp_planted --method aiht --mlist 1 2 5 10 20 50 100 200 500 1000 --slist 1 2 5 10 20 50 100 200 500 1000 --seedlist 0 1 2  --num_epochs 15 --save_results"
```
```
python runner.py --task mlp_planted --method pruning --mlist 1 2 5 10 20 50 100 200 500 1000 --slist 1 2 5 10 20 50 100 200 500 1000 --seedlist 0 1 2  --num_epochs 15 --save_results
```
```
python runner.py --task mlp_planted --method aiht --mlist 2 5 10 20 50 100 200 500 1000 --slist 10 20 50 100 200 500 1000 --seedlist 0 1 2  --num_epochs 15 --num_layers 2 --save_results
```
```
python runner.py --task mlp_planted --method pruning --mlist 2 5 10 20 50 100 200 500 1000 --slist 10 20 50 100 200 500 1000 --seedlist 0 1 2  --num_epochs 15 --num_layers 2 --save_results
```

Figure 2
```
python runner.py --task mlp_planted --method aiht --mlist 1 2 5 10 20 50 100 200 500 1000 --slist 1 2 5 10 20 50 100 200 500 1000 --seedlist 0 1 2  --num_epochs 100 --batch_size 5000 --output_dim 10 --save_results
```
```
python runner.py --task mlp_planted --method pruning --mlist 1 2 5 10 20 50 100 200 500 1000 --slist 1 2 5 10 20 50 100 200 500 1000 --seedlist 0 1 2  --num_epochs 100 --batch_size 5000 --output_dim 10 --save_results
```
```
python runner.py --task mlp_planted --method aiht --mlist 2 5 10 20 50 100 200 500 1000 --slist 10 20 50 100 200 500 1000 --seedlist 0 1 2  --num_epochs 100 --batch_size 5000 --num_layers 2 --output_dim 10 --save_results
```
```
python runner.py --task mlp_planted --method pruning --mlist 2 5 10 20 50 100 200 500 1000 --slist 10 20 50 100 200 500 1000 --seedlist 0 1 2 --num_epochs 100 --batch_size 5000 --num_layers 2 --output_dim 10 --save_results
```

Figure 3
```
python runner.py --task binary_mnist --method aiht --mlist 1 2 5 10 20 50 100 200 500 1000 --slist 1 2 5 10 20 50 100 200 --seedlist 0 1 2 --num_epochs 15 --batch_size 1000 --save_results
```
```
python runner.py --task binary_mnist --method pruning --mlist 1 2 5 10 20 50 100 200 500 1000 --slist 1 2 5 10 20 50 100 200 --seedlist 0 1 2 --num_epochs 15 --batch_size 1000 --save_results
```
```
python runner.py --task mnist --method aiht --mlist 10 20 50 100 200 500 1000 --slist 100 200 500 1000 2000 --seedlist 0 1 2 --num_epochs 50 --batch_size 5000 --save_results
```
```
python runner.py --task mnist --method pruning --mlist 10 20 50 100 200 500 1000 --slist 100 200 500 1000 2000 --seedlist 0 1 2 --num_epochs 50 --batch_size 5000 --save_results
```

Figure 4
```
python runner.py --task inr_mnist --method aiht --mlist 2 5 10 20 50 100 200 --slist 2 5 10 20 50 100 200 500 1000 --seedlist 0 1 2 --num_epochs 100 --save_results
```
```
python runner.py --task inr_mnist --method pruning --mlist 2 5 10 20 50 100 200 --slist 2 5 10 20 50 100 200 500 1000 --seedlist 0 1 2 --num_epochs 100 --save_results
```
```
python runner.py --task inr_mnist --method aiht --mlist 2 5 10 20 50 100 200 --slist 2 5 10 20 50 100 200 500 1000 --seedlist 0 1 2 --num_epochs 100 --num_layers 2 --save_results
```
```
python runner.py --task inr_mnist --method pruning --mlist 2 5 10 20 50 100 200 --slist 2 5 10 20 50 100 200 500 1000 --seedlist 0 1 2 --num_epochs 100 --num_layers 2 --save_results
```

Figure 5
```
python runner.py --task inr_cifar10 --method aiht --mlist 1 2 5 10 20 50 100 200 --slist 5 10 20 50 100 200 500 1000 2000 --seedlist 0 1 2 --num_epochs 100 --save_results
```
```
python runner.py --task inr_cifar10 --method pruning --mlist 1 2 5 10 20 50 100 200 --slist 5 10 20 50 100 200 500 1000 2000 --seedlist 0 1 2 --num_epochs 100 --save_results
```
```
python runner.py --task inr_cifar10 --method aiht --mlist 1 2 5 10 20 50 100 200 --slist 5 10 20 50 100 200 500 1000 2000 --seedlist 0 1 2 --num_epochs 100 --num_layers 2 --save_results
```
```
python runner.py --task inr_cifar10 --method pruning --mlist 1 2 5 10 20 50 100 200 --slist 5 10 20 50 100 200 500 1000 2000 --seedlist 0 1 2 --num_epochs 100 --num_layers 2 --save_results
```

## How to generate heatmap figures

If you run `runner.py` with the `--save_results` flag, it will save a `.npy` file with the performance of each configuration. To convert these log files into heatmap figures, first open `make_figures.py` and edit the task and method at the top. Then run `python make_figures.py`; it will find all log files corresponding to that combination of task and method, and make a heatmap figure corresponding to each log file.
