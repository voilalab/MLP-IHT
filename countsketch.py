# Implementation of a Count Sketch using CuPy
# Using the MurmurHash3 algorithm for hashing, with constants based on https://github.com/rdspring1/Count-Sketch-Optimizers/blob/master/optimizers/sketch.py
import time
import cupy as cp


# Define an elementwise kernel that computes a hash for each input value.
# The kernel takes four inputs: the value to hash, the seeds a and b, and the modulus (range).
# It produces one output: the computed hash.
# Note from https://github.com/gakhov/pdsa/blob/master/pdsa/frequency/count_sketch.pyx
# that this yields a 32-bit hash value. Thus, the length of the counters
# is expected to be smaller or equal to the (2^{32} - 1), since
# we cannot access elements with indexes above this value (unless W is 64 bits).
murmur_hash_kernel = cp.ElementwiseKernel(
    # Input arguments (as C types)
    'T value, W a, W b, W range',
    # Output argument
    'W hash_val',
    # The kernel code (using the finalization steps of MurmurHash3)
    r'''
    // Combine input value with seeds
    unsigned int h = a * value + b;
    // Mix bits: similar to MurmurHash3 finalizer
    h ^= h >> 16;
    h *= 0x85ebca6b;
    h ^= h >> 13;
    h *= 0xc2b2ae35;
    h ^= h >> 16;
    // Take modulo to bring the result into desired range
    hash_val = h % range;
    ''',
    'murmur_hash'
)


class CountSketch:
    def __init__(self, w, d, n, v=4, block_size=100000):
        # Matching the notation from Count-Sketch Optimizers
        # w is the width of the sketch
        # d is the depth of the sketch
        # n is the range of the hash values, the length of the original vector being sketched
        # v is the number of hash functions
        # block_size is the size of the blocks for controlling random seeds for random sign generation
        self.v = v
        self.w = cp.uint32(w)
        self.d = d
        self.n = n
        self.block_size = block_size
        self.total_time = 0
        # Seeds for the first 3 hash functions are borrowed from https://github.com/rdspring1/Count-Sketch-Optimizers/blob/master/optimizers/sketch.py
        # Seeds after the first 3 in each list are random
        self.aa = cp.array([cp.uint32(994443), cp.uint32(4113759), cp.uint32(9171025), cp.uint32(9024893), cp.uint32(12334)])
        self.bb = cp.array([cp.uint32(609478), cp.uint32(2949676), cp.uint32(2171464), cp.uint32(374849), cp.uint32(397372)])
        assert v <= len(self.aa), f'Only {len(self.aa)} hash functions are supported for now. Adding more is easy but requires more magic constants.'
        self.aa = self.aa[0:v]
        self.bb = self.bb[0:v]
        # Initialize the sketch tensor [v, w, d]
        self.tensor = cp.zeros((self.v, self.w, self.d), dtype=cp.float32)
        self.num_updates = 0

    def size(self):
        return self.v * self.w * self.d

    def update_block(self, indices, deltas):
        t0 = time.time()
        self.num_updates += 1
        # indices is a list of indices to update, out of n
        # deltas is a list of values (each length d) to add to the sketch at positions in indices
        if len(deltas.shape) == 1:
            assert self.d == 1
            deltas = deltas[:,cp.newaxis]
        h = cp.zeros((indices.shape[0], self.v), dtype=cp.uint32)  # [batch, self.v]
        for j in range(self.v):
            h[:,j] = murmur_hash_kernel(cp.array(indices), self.aa[j], self.bb[j], self.w)  # [batch,]
        # get random signs blockwise
        block_ids = indices // self.block_size
        unique_block_ids = cp.unique(block_ids)  # This can synchronize the GPU
        for block_id in unique_block_ids:
            cp.random.seed(cp.uint64(block_id.item()))
            # generate signs for the relevant block
            signs = cp.random.choice([-1, 1], size=(self.block_size, self.v))  # [self.block_size, self.v]
            # extract the signs for the relevant indices in this block
            idx_in_block = cp.isin(block_ids, block_id)
            signs = signs[indices[idx_in_block] % self.block_size, :]  # [idx_in_block, self.v]
            # update the sketch tensor for this block
            cp.add.at(self.tensor, (cp.arange(self.v), h[idx_in_block], slice(None)), signs[:, :, cp.newaxis] * deltas[idx_in_block, cp.newaxis, :])  # This is deterministic whereas regular addition is not
        self.total_time += time.time() - t0

    def query_block(self, indices):
        t0 = time.time()
        values = cp.zeros((len(indices), self.d))
        h = cp.zeros((indices.shape[0], self.v), dtype=cp.uint32)  # [batch, self.v]
        for j in range(self.v):
            h[:,j] = murmur_hash_kernel(cp.array(indices), self.aa[j], self.bb[j], self.w)  # [batch,]
        # get random signs blockwise
        block_ids = indices // self.block_size
        unique_block_ids = cp.unique(block_ids)  # This can synchronize the GPU
        for block_id in unique_block_ids:
            cp.random.seed(cp.uint64(block_id.item())) 
            # generate signs for the relevant block
            signs = cp.random.choice([-1, 1], size=(self.block_size, self.v))  # [self.block_size, self.v]
            # extract the signs for the relevant indices in this block
            idx_in_block = cp.isin(block_ids, block_id)
            signs = signs[indices[idx_in_block] % self.block_size, :]  # [idx_in_block, self.v]
            result = self.tensor[cp.arange(self.v), h[idx_in_block], :] * signs[:, :, cp.newaxis]  # [idx_in_block, self.v, self.d]
            values[idx_in_block] = cp.median(result, axis=1)  # [idx_in_block, self.d]
        self.total_time += time.time() - t0
        return values
