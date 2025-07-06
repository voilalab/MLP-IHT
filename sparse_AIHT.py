import cupy as cp
import cupyx
from tqdm import tqdm
import time

from sparse_vector import SparseVector   
from vector_output_utils import VectorOutputMLPWeights
from gaussian_fly_matrix import GaussianFlyMatrix    
from nn_fly_matrix import NNFlyMatrix
from mnist_loader import MNISTLoader, MNISTINRLoader



def support(signal, s):
  # Take the signal and replace the largest s entries (by magnitude) with 1, and the rest with 0 
  idx = cp.argpartition(cp.abs(signal), -s)[-s:]
  x = cp.zeros_like(signal)
  x[idx] = 1
  return x

# Note: this is only actually called during evaluation
def memory_efficient_multiply(A, x, result_s):
    assert type(A) == GaussianFlyMatrix or type(A) == NNFlyMatrix
    assert type(x) == SparseVector
    # If the number of rows is small enough (ie equal to the number of measurements), 
    # operate vectorwise and then keep the top result_s values
    # This is much faster than looping over m, whenever m is small enough we can fit in in memory
    if A.rows < A.cols:
        partial_vector = cp.zeros(A.rows)
        for idx in x.get_indices():
            partial_vector = partial_vector + x.at(idx) * A.column(idx)
        idx = cp.array(range(A.rows))
        idx = idx[cp.abs(partial_vector[idx]) > 0]
        y = SparseVector(n=A.rows, s=result_s, indices=idx, values=partial_vector[idx])
        return y
    # If we make it to this part of the function, then the matrix has been transposed so that rows > cols
    t0 = time.time()
    if type(A) == NNFlyMatrix:
        partial_sum = 0
        # Do the computation blockwise for efficiency
        for block_id in range(A.m):
            start_idx = block_id * A.d
            stop_idx = start_idx + A.d
            x_block = x.dense_block(start_idx=start_idx, stop_idx=stop_idx)  # d
            A_block = A.blockwise_matrix(block_id=block_id)  # [n, d]
            partial_value = cp.dot(A_block, x_block)  # n
            partial_sum = partial_sum + partial_value
        y = SparseVector(n=A.rows, s=result_s, indices=cp.arange(A.rows), values=partial_sum)
        return y
    # The result will be another SparseVector
    # This code will only run if A is a GaussianFlyMatrix
    y = SparseVector(n=A.rows, s=result_s)
    indices = x.get_indices()
    x_sparsedense = x.sparse_dense_version()
    for i in range(A.rows):
        col = A.column(i)
        partial_value = cp.dot(x_sparsedense, col[indices])
        y.update(index=i, value=partial_value)
    return y


# Following https://www.sciencedirect.com/science/article/pii/S0165168411003197?fr=RR-1&ref=cra_js_challenge
def AIHT_step_forGaussian(A, b, x, num_internal_steps=3):
    # In this version, notation is that A is m by n, and n is large
    assert type(A) == GaussianFlyMatrix
    assert type(x) == SparseVector
    assert type(b) == cp.ndarray  # since b is relatively short (length m), do not require sparsity
    assert len(b) == A.rows  # m

    # Compute the adaptive step size
    indices = x.get_indices()
    A_submatrix = A.partial_matrix(column_ids=indices)  # m by s
    x_sparsedense = x.sparse_dense_version()  # s
    Ax_b = A_submatrix @ x_sparsedense - b  # m
    g_sparsedense = A_submatrix.T @ Ax_b  # s
    stepsize = cp.sum(cp.square(g_sparsedense)) / (cp.sum(cp.square(A_submatrix @ g_sparsedense)) + 1e-8)  # scalar
    if len(indices) == 0:
        stepsize = 1e-14

    # Compute gradient simultaneously with taking the step, to lose less precision
    t0 = time.time()
    x_tilde = SparseVector(n=x.n, s=x.s)
    for i in range(A.cols):  # Note: this could be done blockwise for a likely speedup
        col = A.column(i)
        partial_value = cp.dot(Ax_b, col)
        x_tilde.update_block(indices=cp.array([i]), values=cp.array([x.at(i) - stepsize*partial_value]))

    # Search for x_next that reduces the objective relative to x_tilde but retains same sparse support
    x_next = x_tilde
    indices = x_tilde.get_indices()
    x_next = x_tilde.sparse_dense_version()  # s
    A_submatrix = A.partial_matrix(column_ids=indices)  # m by s
    for iter in range(num_internal_steps):
        t0 = time.time()
        Ax_b = A_submatrix @ x_next - b  # m
        if iter == 0:
            fval_tilde = cp.sum(cp.square(Ax_b))
            fval = fval_tilde
        g_sparsedense = A_submatrix.T @ Ax_b  # s
        # Armijo line search
        counter = 0
        x_dummy = x_next
        while True:
            x_dummy = x_next - stepsize * g_sparsedense
            fval_next = cp.sum(cp.square(A_submatrix @ x_dummy - b))
            counter = counter + 1
            if fval_next <= fval - 0.9 * stepsize * cp.sum(cp.square(g_sparsedense)):
                fval = fval_next
                x_next = x_dummy
                break
            stepsize = stepsize / 2
            if counter > 10: # Give up eventually so we don't sit in an infinite loop (this shouldn't happen in practice)
                print(f'linesearch failed')
                x_next = x_tilde.sparse_dense_version()
                fval = fval_tilde
                break
    return SparseVector(n=x.n, s=x.s, indices=indices, values=x_next)


time_making_blockwise_matrix = 0
time_update = 0
time_in_loop = 0
time_culling_updates = 0
time_making_submatrix = 0
time_in_partial_matrix = 0
time_linesearch = 0
time_stepsize = 0
time_dot = 0
time_block = 0
time_step = 0
time_construction = 0



# Following https://www.sciencedirect.com/science/article/pii/S0165168411003197?fr=RR-1&ref=cra_js_challenge
def AIHT_step_vectoroutput(A, b, x, num_internal_steps=0, epoch=0, crossentropy=False):
    # In this notation, A is n by md, and md is large
    # d is the dimension of each example, n is the number of examples, and m is the number of hidden neurons
    # b is a matrix of labels, of shape n by k where k is the output dimension
    assert type(A) == NNFlyMatrix
    assert type(x) == SparseVector
    assert type(b) == cp.ndarray  # since b is relatively short, do not require sparsity
    assert len(b) == A.rows  # n
    global time_making_blockwise_matrix
    global time_update
    global time_in_loop
    global time_culling_updates
    global time_making_submatrix
    global time_in_partial_matrix
    global time_linesearch
    global time_stepsize
    global time_dot
    global time_block
    global time_construction

    w = VectorOutputMLPWeights(W=x, d=A.d, m=A.m, k=b.shape[1])

    # Compute the adaptive step size
    t0 = time.time()
    indices = w.W1_md().get_indices()
    A_submatrix, partialmattime = A.partial_matrix(column_ids=indices)  # [n, s]
    time_in_partial_matrix += partialmattime
    ta = time.time()
    time_making_blockwise_matrix += ta - t0
    x_sparsedense = w.W1_W2_sparsedense(row_ids=indices)  # [s, k]
    Ax = A_submatrix @ x_sparsedense  # [n, k]
    if crossentropy:
        Ax_b = (cupyx.scipy.special.softmax(Ax, axis=1) - b) / A.n # Normalizing by n is very important here, at least if using fixed stepsizes
    else:
        Ax_b = (Ax - b) / A.n  # [n, k] 
    td = time.time()
    time_stepsize += td - ta

    # Do the computation blockwise for efficiency
    x_tilde = SparseVector(n=x.n, s=x.s, indices=x.indices, values=x.values)
    x_tilde.use_countsketch = x.use_countsketch
    x_tilde.countsketch = x.countsketch
    w_tilde = VectorOutputMLPWeights(W=x_tilde, d=A.d, m=A.m, k=b.shape[1])
    Ax_check = 0
    for block_id in range(A.m):
        ta = time.time()
        A_block = A.blockwise_matrix(block_id=block_id)  # [n, d]  # this is very cheap for small m, but gets large if m is large
        t0 = time.time()
        time_block += t0-ta
        partial_value = cp.matmul(A_block.T, Ax_b)  # [d, k]
        
        # use chain rule to get gradients for W1 and W2
        w1_block, w2_block = w.get_blocks(block_id=block_id)
        time_dot += time.time() - t0
        ta = time.time()
        # The problem is this chain rule produces zero gradients to both w1 and w2 since both are starting at 0. 
        # This is a hack around that, using Xavier initialization
        if cp.max(w1_block) == 0 and epoch == 0:
            cp.random.seed(block_id)
            w1_block = cp.random.uniform(low=-1/cp.sqrt(A.d*A.m), high=1/cp.sqrt(A.d*A.m), size=len(w1_block))
        if cp.max(w2_block) == 0 and epoch == 0:
            cp.random.seed(block_id)
            w2_block = cp.random.uniform(low=-1/cp.sqrt(A.m*b.shape[1]), high=1/cp.sqrt(A.m*b.shape[1]), size=len(w2_block))
        w1_grad = cp.dot(partial_value, w2_block)  # d
        w2_grad = cp.dot(partial_value.T, w1_block)  # k
        basestepsize = 0.005
        if x.s < 6 * A.m**0.536:  
            basestepsize = 2  # 0.5  # small s cases benefit from larger stepsizes
        if crossentropy:
            basestepsize = 0.001
        alpha1 = basestepsize
        alpha2 = basestepsize
        tb = time.time()
        time_making_submatrix += tb - ta  # not really making submatrix, actually time to compute the gradient
            
        w1_values = w1_block - alpha1 * w1_grad
        w2_values = w2_block - alpha2 * w2_grad
        time_construction += time.time() - tb  # not really construction, actually for apply gradient to take a step
        t0, t1 = w_tilde.update_W1_block(block_id=block_id, values=w1_values, deltas=-alpha1*w1_grad)
        time_culling_updates += t0
        time_update += t1
        t0, t1 = w_tilde.update_W2_block(block_id=block_id, values=w2_values, deltas=-alpha2*w2_grad)
        time_culling_updates += t0
        time_update += t1
    tb = time.time()
    time_in_loop += tb - td
    # Local refinement (acceleration) doesn't seem to help, probably because it's a nonconvex matrix product
    return w_tilde.W


# Following https://www.sciencedirect.com/science/article/pii/S0165168411003197?fr=RR-1&ref=cra_js_challenge
def AIHT_step(A, b, x, num_internal_steps=3):
    # In this notation A is n by md, and md is large
    # d is the dimension of each example, n is the number of examples, and m is the number of hidden neurons
    assert type(A) == NNFlyMatrix
    assert type(x) == SparseVector
    assert type(b) == cp.ndarray  # since b is relatively short, do not require sparsity
    assert len(b) == A.rows  # n
    global time_making_blockwise_matrix
    global time_update
    global time_in_loop
    global time_culling_updates
    global time_making_submatrix
    global time_in_partial_matrix
    global time_linesearch
    global time_stepsize
    global time_dot
    global time_block
    global time_construction
    
    # Compute the adaptive step size
    t0 = time.time()
    indices = x.get_indices()
    A_submatrix, partialmattime = A.partial_matrix(column_ids=indices)  # n by s
    time_in_partial_matrix += partialmattime
    ta = time.time()
    time_making_blockwise_matrix += ta - t0
    x_sparsedense = x.sparse_dense_version()  # s
    Ax_b = A_submatrix @ x_sparsedense - b  # n
    g_sparsedense = A_submatrix.T @ Ax_b  # s
    stepsize = cp.sum(cp.square(g_sparsedense)) / (cp.sum(cp.square(A_submatrix @ g_sparsedense)) + 1e-8)
    if len(indices) == 0:
        stepsize = 1e-14
    x_tilde = SparseVector(n=x.n, s=x.s)
    x_tilde.use_countsketch = x.use_countsketch
    x_tilde.countsketch = x.countsketch
    # Do the computation blockwise for efficiency
    td = time.time()
    time_stepsize += td - ta
    for block_id in range(A.m):
        ta = time.time()
        block = A.blockwise_matrix(block_id=block_id)  # [n, d]  # this is very cheap for small m, but gets large if m is large
        t0 = time.time()
        time_block += t0-ta
        partial_value = cp.dot(Ax_b, block)  # d
        time_dot += time.time() - t0
        start_idx = block_id * A.d
        stop_idx = start_idx + A.d
        indices = cp.arange(start_idx, stop_idx)
        values = x.dense_block(start_idx=start_idx, stop_idx=stop_idx) - stepsize*partial_value
        t0, t1 = x_tilde.update_block(indices=indices, values=values, deltas=-stepsize*partial_value)
        time_culling_updates += t0
        time_update += t1
    tb = time.time()
    time_in_loop += tb - td

    # Search for x_next that reduces the objective relative to x_tilde but retains same sparse support
    indices = x_tilde.get_indices()
    x_next = x_tilde.sparse_dense_version()  # s
    A_submatrix, partialmattime = A.partial_matrix(column_ids=indices)  # n by s
    time_in_partial_matrix += partialmattime
    tc = time.time()
    time_making_submatrix += tc - tb
    for iter in range(num_internal_steps):
        t0 = time.time()
        Ax_b = A_submatrix @ x_next - b  # n
        if iter == 0:
            fval_tilde = cp.sum(cp.square(Ax_b))
            fval = fval_tilde
        g_sparsedense = A_submatrix.T @ Ax_b  # s
        # Armijo line search
        counter = 0
        while True:
            x_dummy = x_next - stepsize * g_sparsedense  
            fval_next = cp.sum(cp.square(A_submatrix @ x_dummy - b))
            counter = counter + 1
            if fval_next <= fval - 0.9 * stepsize * cp.sum(cp.square(g_sparsedense)):
                fval = fval_next
                x_next = x_dummy
                break
            stepsize = stepsize / 2
            if counter > 10: # Give up eventually so we don't sit in an infinite loop
                x_next = x_tilde.sparse_dense_version()
                fval = fval_tilde
                break
    tl = time.time()
    time_linesearch += tl - tc
    result = SparseVector(n=x.n, s=x.s, indices=indices, values=x_next)
    result.use_countsketch = x.use_countsketch
    result.countsketch = x.countsketch
    time_construction += time.time() - tl
    return result


def AIHT(A, b_dense, result_s, num_steps, random_h, SCP_rate=1, crossentropy=False):  # SCP 5 was best for binary mnist before fixing treap
    if len(b_dense.shape) > 1:
        x = SparseVector(n=A.d*A.m + A.m*b_dense.shape[1], s=result_s, use_countsketch=True)
    else:
        # Fuse the last layer weights if we have scalar output
        x = SparseVector(n=A.cols, s=result_s, use_countsketch=False)  # countsketch is not helpful for fused weights
    for step in tqdm(range(num_steps)):
        if type(A) == GaussianFlyMatrix:
            x = AIHT_step_forGaussian(A, b_dense, x, num_internal_steps=3)
        else:
            if len(b_dense.shape) > 1:
                x = AIHT_step_vectoroutput(A, b_dense, x, num_internal_steps=0, epoch=step, crossentropy=crossentropy)
    
            else:
                x = AIHT_step(A, b_dense, x, num_internal_steps=10)  # for INR_MNIST and INR_CIFAR10, 10 internal steps is typically best
        if not random_h and type(A) == NNFlyMatrix and step > 0 and step % SCP_rate == 0:  
            if len(b_dense.shape) > 1:
                A.set_hs(x.extract_block(start_idx=0, stop_idx=A.m*A.d))
            else:
                A.set_hs(x)
    assert x.s <= result_s; f'x.s is {x.s} but result_s is {result_s}'
    return x



def stochastic_AIHT(m, train_loader, result_s, num_epochs, random_h, SCP_rate=1, seed=0, crossentropy=False):
    # train_loader is a torch DataLoader that iterates through the training data
    # This can be used instead of full-batch AIHT for datasets that are too large, and in some cases it improves accuracy
    if len(train_loader) == 1:
        print(f'for full-batch optimization please use AIHT rather than stochastic_AIHT; it will be faster')
    old_x = None
    x = None
    global time_step
    num_batches_per_epoch = len(train_loader)
    time_making_matrix = 0
    time_in_seth_and_extractblock = 0
    time_in_step = 0
    print(f'num batches per epoch is {num_batches_per_epoch}, num epochs is {num_epochs}')
    tloader = 0
    toutside = time.time()
    for epoch in tqdm(range(num_epochs)):
        for step, data in enumerate(train_loader):
            tstart = time.time()
            # Every data instance is an input + label pair
            inputs, labels = data
            labels = cp.asarray(labels)
            # Generate the A matrix for this batch. It's shape is n by md, 
            # where n is the batch size, m is the hidden dimension, and d is the input dimension
            t0 = time.time()
            A = NNFlyMatrix(m=m, X=cp.asarray(inputs), cache_size=result_s, seed=seed)
            t1 = time.time()
            time_making_matrix += t1 - t0
            if not random_h and old_x is not None:
                if len(labels.shape) > 1:
                    A.set_hs(old_x.extract_block(start_idx=0, stop_idx=A.m*A.d))
                else:
                    A.set_hs(old_x)
            t2 = time.time()
            time_in_seth_and_extractblock += t2 - t1
            if x is None:
                if len(labels.shape) > 1:
                    x = SparseVector(n=A.d*A.m + A.m*labels.shape[1], s=result_s, use_countsketch=True)
                else:
                    # Fuse the last layer weights if we have scalar output
                    x = SparseVector(n=A.cols, s=result_s, use_countsketch=False)  # countsketch is unhelpful for scalar outputs
            t0 = time.time()

            if len(labels.shape) > 1:
                x = AIHT_step_vectoroutput(A, labels, x, num_internal_steps=0, epoch=step, crossentropy=crossentropy)
            else:
                x = AIHT_step(A, cp.asarray(labels), x, num_internal_steps=10)
            time_in_step += time.time() - t0
            time_step += time.time() - t0
            tloader += time.time() - tstart
            if not random_h and epoch > 0 and (step + epoch*num_batches_per_epoch) % SCP_rate == 0:   # It does seem marginally better to stay convex for the first full epoch, but it's not a huge difference
                old_x = x
    tloader = time.time() - toutside - tloader
    assert x.s <= result_s; f'x.s is {x.s} but result_s is {result_s}'
    print(f'time in dataloader loop={tloader}, time in step = {time_in_step}, time_making_matrix={time_making_matrix}, time_in_seth_and_extractblock={time_in_seth_and_extractblock}')
    print(f'total time in step = {time_step}, total time in partial_matrix_loop={time_in_partial_matrix}, time_making_blockwise_matrix={time_making_blockwise_matrix}, time_stepsize={time_stepsize}, time_block={time_block}, time_dot={time_dot}, time_culling_updates={time_culling_updates}, time_update={time_update}, time_in_loop={time_in_loop}, time_making_submatrix={time_making_submatrix}, time_linesearch={time_linesearch}, time_construction={time_construction}')
    return x


def evalBinaryAcc(test_loader, x_hat, m, convex, seed=0):
    acc_sum = 0
    count = 0
    for step, data in enumerate(test_loader):
        # Every data instance is an input + label pair
        inputs, labels = data
        A = NNFlyMatrix(m=m, X=cp.asarray(inputs), seed=seed)
        if not convex:
            A.set_hs(x_hat)
        prediction = memory_efficient_multiply(A=A, x=x_hat, result_s=A.rows).dense_version()
        prediction[prediction < 0.5] = 0
        prediction[prediction >= 0.5] = 1
        acc_sum += cp.mean(prediction == cp.asarray(labels))
        count += 1
    return acc_sum / count


# Multiclass test accuracy
def evalAcc(test_loader, x_hat, m, convex, seed=0):
    acc_sum = 0
    count = 0
    w_hat = None
    for step, data in enumerate(test_loader):
        # Every data instance is an input + label pair
        # Assume labels are one-hot
        inputs, labels = data
        A = NNFlyMatrix(m=m, X=cp.asarray(inputs), seed=seed)
        if w_hat is None:
            w_hat = VectorOutputMLPWeights(W=x_hat, d=A.d, m=A.m, k=labels.shape[1])
        if not convex:
            A.set_hs(w_hat.W1_md())
        prediction = w_hat.A_multiply_dense(A=A)  # [batch_size, num_classes]
        prediction = cp.argmax(prediction, axis=1)
        label = cp.argmax(cp.array(labels), axis=1)
        acc_sum += cp.mean(prediction == label)
        count += 1
    return acc_sum / count


# Vector output test MSE
def evalMSE(test_loader, x_hat, m, convex, seed=0):
    mse_sum = 0
    count = 0
    w_hat = None
    for _, data in enumerate(test_loader):
        # Every data instance is an input + label pair
        inputs, labels = data
        A = NNFlyMatrix(m=m, X=cp.asarray(inputs), seed=seed)
        if w_hat is None:
            w_hat = VectorOutputMLPWeights(W=x_hat, d=A.d, m=A.m, k=labels.shape[1])
        if not convex:
            A.set_hs(w_hat.W1_md())
        prediction = w_hat.A_multiply_dense(A=A)  # [batch_size, num_classes]
        mse_sum += cp.mean(cp.square(prediction - cp.array(labels)))
        count += 1
    return mse_sum / count


# This actually runs both binary and multiclass classification experiments, depending on the shape of train_labels
def binaryExperiment(m, s, train_loader, test_loader, convex, num_epochs, seed=0, crossentropy=False):
    train, train_labels = next(iter(train_loader))
    if len(train_loader) == 1:
        # full-batch optimization is faster using raw data
        A = NNFlyMatrix(m=m, X=cp.array(train), seed=seed)
        x_hat = AIHT(A=A, b_dense=cp.array(train_labels), result_s=s, num_steps=num_epochs, random_h=convex, crossentropy=crossentropy)
    else:
        x_hat = stochastic_AIHT(m=m, train_loader=train_loader, result_s=s, num_epochs=num_epochs, random_h=convex, seed=seed, crossentropy=crossentropy)
    if len(train_labels.shape) == 1:
        test_acc = evalBinaryAcc(test_loader=test_loader, x_hat=x_hat, m=m, convex=convex, seed=seed)
    else:
        test_acc = evalAcc(test_loader=test_loader, x_hat=x_hat, m=m, convex=convex, seed=seed)
    print(f'test accuracy is {test_acc * 100}% with m={m} and s={s}')
    return test_acc, x_hat


def INRExperiment(m, s, train_loader, convex, num_epochs=15, seed=0, max_label=1):
    t0 = time.time()
    train, train_labels = next(iter(train_loader))
    if len(train_loader) == 1:
        # full-batch optimization is faster using raw data
        A = NNFlyMatrix(m=m, X=cp.array(train), seed=seed)
        x_hat = AIHT(A=A, b_dense=cp.array(train_labels), result_s=s, num_steps=num_epochs, random_h=convex, crossentropy=False)
    else:
        x_hat = stochastic_AIHT(m, train_loader, result_s=s, num_epochs=num_epochs, random_h=convex, crossentropy=False)  # HERE is called for vector output MLP
    print(f'solve took {time.time() - t0} seconds')
    if len(train_labels.shape) > 1:
        # Vector output case
        mse = evalMSE(train_loader, x_hat, m, convex, seed=seed)
        psnr = 20*cp.log10(max_label) - 10*cp.log10(mse)
        print(f'psnr is {psnr} with m={m} and s={s}')
        return psnr, x_hat
    mses = []
    for _, data in enumerate(train_loader):
        # Every data instance is an input + label pair
        inputs, labels = data
        # Generate the A matrix for this batch. It's shape is n by md, 
        # where n is the batch size, m is the hidden dimension, and d is the input dimension
        A = NNFlyMatrix(m=m, X=cp.asarray(inputs), cache_size=s, seed=seed)
        if not convex:
            A.set_hs(x_hat)
        prediction = memory_efficient_multiply(A=A, x=x_hat, result_s=A.rows).dense_version()
        mses.append(cp.mean(cp.square(prediction - cp.asarray(labels))))
    psnr = 20*cp.log10(max_label) - 10*cp.log10(cp.mean(cp.asarray(mses)))
    print(f'psnr is {psnr} with m={m} and s={s}')
    return psnr, x_hat


def GaussianExperiment(n, s, m, num_steps, seed=0):
    cp.random.seed(seed)
    x = SparseVector(n=n, s=s, indices=cp.array([a.item() for a in cp.random.choice(a=n, size=s, replace=False)]), values=cp.array([1]*s))
    A = GaussianFlyMatrix(m=m, n=n, seed=seed)
    A_submatrix = A.partial_matrix(column_ids=x.get_indices())  # m by s
    x_sparsedense = x.sparse_dense_version()  # length s
    y = A_submatrix @ x_sparsedense  # length m

    x_star = x.dense_version()
    t0 = time.time()
    x_hat = AIHT(A=A, b_dense=y, result_s=s, num_steps=num_steps, random_h=True).dense_version()
    print(f'x_star is {x_star}, x_hat is {x_hat}')
    print(f'solve took {time.time() - t0} seconds')
    l2_error = cp.linalg.norm(x_hat - x_star) / cp.linalg.norm(x_star)
    support_error = cp.linalg.norm(support(x_hat, s=s) - support(x_star, s=s), ord=1) / (2*cp.linalg.norm(support(x_star, s=s), ord=1))
    print(f'm={m}, n={n}, s={s}: l2 error is {l2_error}, support error is {support_error}')
    return l2_error