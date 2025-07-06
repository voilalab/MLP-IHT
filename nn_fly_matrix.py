import cupy as cp
import time


# This is the A matrix corresponding to a convex (or nonconvex) ReLU neural network with scalar output
class NNFlyMatrix:
    def __init__(self, m, X, cache_size=0, seed=0):
        # d is the dimension of a data point (e.g. for MNIST this is 28*28+1=785 with bias or 28*28=784 without bias)
        # n is the number of data points (e.g. for binary MNIST this is 10k since each digit has 5k)
        # m is the hidden dimension of the NN
        # A is n by md
        self.m = m
        self.X = X  # n by d, data matrix
        (self.n, self.d) = X.shape
        self.hs = None
        self.rows = self.n
        self.cols = self.m * self.d
        cp.random.seed(seed)
        self.seedoffset = cp.random.randint(low=0, high=self.cols).item()
        if seed == 0:
            self.seedoffset = 0  # special case for clean testing
        if self.rows > self.cols:  # measurement matrix should be rectangular (compressed sensing)
            print(f'WARNING: not in compressed sensing mode. Recommend checking dimensions.')
        
    def set_hs(self, hs):
        if hs is None:
            return
        assert hs.n == self.cols
        self.hs = hs
    
    # This takes almost as long as blockwise_matrix, so it's generally more efficient to use blockwise_matrix
    def column(self, j):  # if h is nonrandom this corresponds to a nonconvex NN
        if hasattr(j, 'item'):
            j = j.item()
        block_id = j // self.d
        start_idx = block_id * self.d
        stop_idx = start_idx + self.d
        # First, check if hs is None; if so then generate h
        if self.hs is None:
            cp.random.seed(block_id + self.seedoffset)
            h = cp.random.normal(size=self.d)  # d 
        # Select the appropriate h for this block
        else:
            h = self.hs.dense_block(start_idx=start_idx, stop_idx=stop_idx)
        # Compute the indicator and matrix-vector product
        indicator = cp.matmul(self.X, h) > 0  # n
        return indicator * self.X[:,j % self.d]

    # Operate blockwise, since the matrix is actually structured blockwise
    # Each block is n by d, and there are m blocks
    def blockwise_matrix(self, block_id):
        start_idx = block_id * self.d
        stop_idx = start_idx + self.d
        # First, check if hs is None; if so then generate h
        if self.hs is None:
            cp.random.seed(block_id + self.seedoffset)
            h = cp.random.normal(size=self.d)  # d 
        # Select the appropriate h for this block
        else:
            h = self.hs.dense_block(start_idx=start_idx, stop_idx=stop_idx)
        # Compute the shared indicator vector Xh
        indicator = cp.greater(cp.dot(self.X, h), 0)  # [n]
        # Compute the elementwise product for this submatrix
        return indicator[:,cp.newaxis] * self.X  # [n, d]

    # Generate the m by s submatrix comprised of the given column ids 
    # Note that if the column_ids correspond to a contiguous block, it will be slightly more efficient to use blockwise_matrix
    def partial_matrix(self, column_ids):
        mat = cp.zeros((self.rows, len(column_ids))) 
        if len(column_ids) < 1:
            return mat, 0
        t0 = time.time()
        block_ids = column_ids // self.d
        unique_block_ids = cp.unique(block_ids)  # This can synchronize the GPU
        for block_id in unique_block_ids:
            block = self.blockwise_matrix(block_id=block_id.item())
            col_idx = cp.isin(block_ids, block_id)
            mat[:,col_idx] = block[:,column_ids[col_idx] - block_id * self.d]
        return mat, time.time() - t0
