import matplotlib.pyplot as plt
import os
import glob
import numpy as np


# Choose the task and method for which you want to create heatmaps
task = 'MNIST' # Choose one of ['MNIST', 'BINARY_MNIST', 'BINARY_CIFAR10', 'MLP_PLANTED', 'INR_MNIST', 'INR_CIFAR10', 'GAUSSIAN_SPARSE']
method = 'aiht' # Either 'aiht' or 'pruning'
show_title = False



LABELSIZE = 14
plt.rcParams.update({'font.size': 18, 'xtick.labelsize': LABELSIZE, 'ytick.labelsize': LABELSIZE})

filenames = glob.glob(pathname=os.path.join(task, os.path.join(method, '*.npy')))

def extract(filename):
    print(filename)
    acc = np.load(filename)
    mlist = [int(m) for m in filename.split('m[')[1].split(']')[0].split(', ')]
    slist = [int(s) for s in filename.split('s[')[1].split(']')[0].split(', ')]
    seedlist = [0]
    hidden = 1
    if 'hidden' in filename:
        hidden = int(filename.split('hidden')[1].split('_')[0])
    if 'seed' in filename:
        seedlist = [int(s) for s in filename.split('seed[')[1].split(']')[0].split(', ')]
    bs = filename.split('_bs')[1].split('_')[0]
    if bs=='0':
        bs = 'full'
    seed = 'seed'
    if len(seedlist) > 1:
        seed = 'seeds'
    title = f'{task}_hidden{hidden}_{method}_batch{bs}_{len(seedlist)}{seed}'
    return acc, slist, mlist, seedlist, title


def make_heatmap(filename):
    print(f'processing {filename}')
    acc, slist, mlist, seedlist, title = extract(filename)
    plt.figure()
    if len(acc.shape) == 2:
        plt.imshow(acc)
    else:
        plt.imshow(np.mean(acc, axis=2))
    plt.ylabel("m")
    plt.xlabel("s")
    plt.xticks(ticks=range(len(slist)), labels=[str(s) for s in slist], rotation=45)
    plt.yticks(ticks=range(len(mlist)), labels=[str(m) for m in mlist])
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=LABELSIZE)
    if 'PLANTED' in task:
        cbar.mappable.set_clim(-20, 50)
    elif 'INR' in task:
        cbar.mappable.set_clim(0, 50)
    elif 'BINARY' in task:
        cbar.mappable.set_clim(0.5, 1)
    else:
        cbar.mappable.set_clim(0.1, 1)
    if show_title:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(filename[:-3]+'jpg')

for filename in filenames:
    make_heatmap(filename)