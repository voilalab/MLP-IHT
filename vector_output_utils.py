import cupy as cp
from sparse_vector import SparseVector      
from nn_fly_matrix import NNFlyMatrix
from gaussian_fly_matrix import GaussianFlyMatrix  
import time
import scipy.sparse as sparse


# An interface between a traditional 2-layer MLP with vector inputs and outputs
# and the matrix-vector product formulation used for AIHT
class VectorOutputMLPWeights():
    def __init__(self, W, d, m, k):
        assert type(W) is SparseVector
        # d = input dimension
        # m = hidden dimension
        # k = output dimension
        self.W = W  # concatenated sparse vector of weights, of length md + mk
        self.d = d
        self.m = m
        self.k = k
        assert W.n == d * m + m * k

    def print_sparsity_levels(self):
        w1 = self.W.extract_block(start_idx=0, stop_idx=self.m*self.d)
        w2 = self.W.extract_block(start_idx=self.m*self.d, stop_idx=self.m*(self.d + self.k))
        print(f'w1 has {w1.numentries/(self.m*self.d)} fraction of nonzero entries')
        print(f'w2 has {w2.numentries / (self.k*self.m)} fraction of nonzero entries')
        print(f'overall, W has {w1.numentries + w2.numentries} nonzeros, while s is {self.W.s}')

    # Extract just the weights of the hidden layer, as a sparse vector of length md
    # This extracted sparse vector is used to make the A matrix
    def W1_md(self):
        return self.W.extract_block(start_idx=0, stop_idx=self.m*self.d)
    
    # Extract just the weights of the output layer, as a sparse vector of length mk
    def W2_mk(self):
        return self.W.extract_block(start_idx=self.m*self.d, stop_idx=self.W.n)
    
    def get_blocks(self, block_id):
        # Get the corresponding blocks of W1 and W2, length d and k respectively
        if hasattr(block_id, 'item'):
            block_id = block_id.item()
        w1_block = self.W.dense_block(start_idx=block_id*self.d, stop_idx=(block_id+1)*self.d)
        w2_start_idx = self.d * self.m
        w2_block = self.W.dense_block(start_idx=w2_start_idx + block_id*self.k, stop_idx=w2_start_idx + (block_id+1)*self.k)
        return w1_block, w2_block
    
    def update_W1_block(self, block_id, values, deltas=None):
        indices = cp.arange(block_id*self.d, (block_id+1)*self.d)
        t0, t1 = self.W.update_block(indices=indices, values=values, deltas=deltas)
        return t0, t1

    def update_W2_block(self, block_id, values, deltas=None):
        w2_start_idx = self.d * self.m
        indices = cp.arange(w2_start_idx + block_id*self.k, w2_start_idx + (block_id+1)*self.k)
        t0, t1 = self.W.update_block(indices=indices, values=values, deltas=deltas)
        return t0, t1

    def W1_W2_block(self, block_id):
        # Extract corresponding blocks of W1 and W2 and take their outer product as a dense matrix
        w1_block, w2_block = self.get_blocks(block_id=block_id)
        return cp.outer(w1_block, w2_block)  # [d, k]
    
    def W1_W2_sparsedense(self, row_ids):
        # The full concatenation of outer products of W1 and W2 blocks is shape md by k
        # Here we only want the rows that contain a nonzero entry
        # so the shape will be (at most s) by k
        mat = cp.zeros((len(row_ids), self.k)) 
        if len(row_ids) < 1:
            return mat
        block_ids = row_ids // self.d  # there are m blocks in the full matrix
        unique_block_ids = cp.unique(block_ids)  # This can synchronize the GPU
        for block_id in unique_block_ids:
            block = self.W1_W2_block(block_id=block_id.item())
            row_idx = cp.isin(block_ids, block_id)
            mat[row_idx,:] = block[row_ids[row_idx] - block_id * self.d, :]
        return mat

    def A_multiply_dense(self, A):
        assert type(A) == NNFlyMatrix
        assert A.m == self.m
        assert A.d == self.d
        partial_sum = 0
        # Do the computation blockwise for efficiency
        for block_id in range(A.m):
            x_block = self.W1_W2_block(block_id=block_id)  # [d, k]
            A_block = A.blockwise_matrix(block_id=block_id)  # [n, d]
            partial_value = cp.matmul(A_block, x_block)  # [n, k]
            partial_sum = partial_sum + partial_value
        return partial_sum  # dense matrix of shape [n, k]