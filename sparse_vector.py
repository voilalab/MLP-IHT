import cupy as cp
import time
from countsketch import CountSketch


nonzero_trues = cp.ElementwiseKernel(
    'T x, bool b',
    'bool y',
    'y = b & (x != 0)?1:0',
    'nonzero_trues'
)

clamp_kernel = cp.ElementwiseKernel(
    'T x, T l, T u',
    'bool y',
    'y = (x >= l) & (x < u)',
    'clamp_kernel'
)


class SparseVector:
    def __init__(self, n, s, indices=cp.array([]), values=cp.array([]), use_countsketch=False):
        self.n = n
        self.s = s
        if n < s:
            print(f'n={n} which is less than s={s}. Setting s=n which amounts to no sparsity.')
            self.s = self.n
        assert len(indices) == len(values)
        assert len(indices) <= self.s
        self.numentries = len(indices)
        if self.numentries > 0:
            assert max(indices) < self.n
        # Implement the SparseVector as two parallel cupy arrays for indices and values
        self.indices = cp.ones(self.s, dtype=int) * -1  # index -1 means unoccupied
        self.values = cp.zeros(self.s)
        self.indices[0:self.numentries] = indices
        self.values[0:self.numentries] = values
        # Optionally, use a count sketch for noisy but global importance weighting
        self.use_countsketch = use_countsketch
        self.countsketch = None
        if self.use_countsketch:
            scale = cp.log(self.n / self.s)
            if scale < 1:
                scale = 1
            print(f'Using a countsketch for importance weighting: s is {self.s}, n is {self.n}, s*log(n/s) is {int(self.s*cp.log(self.n/self.s))}, using width {int(self.s * scale)}')
            self.countsketch = CountSketch(w=int(self.s * scale), d=1, n=self.n)

    def get_indices(self):
        indices = self.indices
        idx = indices >= 0
        return indices[idx]

    # Return the value at the given index
    def at(self, index):
        idx = cp.nonzero(self.indices == index)[0]
        if len(idx) == 0:
            return 0
        return self.values[idx[0]]

    # Extract the subvector [start_idx, stop_idx) as a sparse vector
    def extract_block(self, start_idx, stop_idx):
        indices = self.indices
        values = self.values
        idx = indices >= start_idx
        indices = indices[idx]
        values = values[idx]
        idx = indices < stop_idx
        indices = indices[idx]
        values = values[idx]
        return SparseVector(n=stop_idx - start_idx, s=self.s, indices=indices - start_idx, values=values)

    def update_block(self, indices, values, deltas=None):
        if self.use_countsketch:
            # update the count sketch first so that we can use it as importance weights when updating the sparse vector
            assert deltas is not None
            self.countsketch.update_block(indices=indices, deltas=deltas)
        t0 = time.time()
        # First update any values that are already present in the vector
        _, idxtoupdate, newidx = cp.intersect1d(self.indices, indices, assume_unique=True, return_indices=True) 
        self.values[idxtoupdate] = values[newidx]
        # Remove these from the list of updates now that we have updated them
        newidx = cp.in1d(indices, self.indices, invert=True)  
        # Then consider the remaining new updates and filter any that are zero-valued
        newidx = cp.nonzero(nonzero_trues(values, newidx))
        indices = indices[newidx]
        values = values[newidx]
        t2 = time.time()
        self.indices = cp.concatenate([self.indices, indices])
        self.values = cp.concatenate([self.values, values])
        # Remove indices that are negative
        indices = cp.argwhere(self.indices >= 0)[:,0]
        self.indices = self.indices[indices]
        self.values = self.values[indices]
        # Remove redundant indices
        self.indices, indices = cp.unique(self.indices, return_index=True)
        self.values = self.values[indices]
        if len(self.indices) > self.s:
            if self.use_countsketch:
                magnitude_ordering = cp.argsort(cp.abs(self.countsketch.query_block(self.indices))[:,0])[len(self.indices) - self.s:]  # if d for the countsketch is >1 this will lose information
            else:
                magnitude_ordering = cp.argsort(cp.abs(self.values))[len(self.indices)-self.s:]
            self.indices = self.indices[magnitude_ordering]
            self.values = self.values[magnitude_ordering]
        # Sort again so that the indices are in order
        ordering = cp.argsort(self.indices)
        self.indices = self.indices[ordering]
        self.values = self.values[ordering]
        return t2-t0, time.time() - t2

    def dense_version(self):
        arr = cp.zeros(self.n)
        arr[self.indices] = self.values
        return arr
    
    # Return a dense vector consisting of the indices [start_idx, stop_idx)
    def dense_block(self, start_idx, stop_idx):
        arr = cp.zeros(stop_idx - start_idx)
        idx = clamp_kernel(self.indices, start_idx, stop_idx)
        arr[self.indices[idx] - start_idx] = self.values[idx]
        return arr
        
    # Return a dense vector that is only the nonzero entries in order, but losing precise index information
    def sparse_dense_version(self):
        idx = self.indices >= 0
        values = self.values[idx]
        return values
