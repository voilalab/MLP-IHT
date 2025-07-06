import cupy as cp

# This is a Gaussian matrix that we can generate on the fly entry by entry (or really column by column)
class GaussianFlyMatrix:
    def __init__(self, m, n, seed=0):
        # m by n Gaussian matrix
        self.m = m
        self.n = n
        self.d = n // m
        self.transpose = False
        self.rows = self.m
        self.cols = self.n
        if self.rows > self.cols:  # expect measurement matrix should be rectangular (compressed sensing)
            print(f'WARNING: not in compressed sensing mode. Recommend checking dimensions.')
        cp.random.seed(seed)
        self.seedoffset = cp.random.randint(low=0, high=self.cols).item()

    def T(self):
        self.transpose = not self.transpose
        temp = self.rows
        self.rows = self.cols
        self.cols = temp
    
    def column(self, j):
        # Columns are always generated in the original (non-transposed) shape
        # Because we assume m << n, so we can generate an m-vector but not an n-vector
        if hasattr(j, "item"):
            j = j.item()
        cp.random.seed(j + self.seedoffset)
        return cp.random.normal(size=self.m)

    # Generate the m by s submatrix comprised of the given column ids    
    def partial_matrix(self, column_ids):
        # Never generate a dense partial matrix that is larger than measurements by measurements
        if len(column_ids) > self.rows:
            print(f'WARNING: CANNOT GENERATE A MATRIX THIS LARGE; TRUNCATING columns from {len(column_ids)} to {self.rows}')
            column_ids = column_ids[0:self.rows]
        mat = cp.zeros((self.rows, len(column_ids))) 
        for key, id in enumerate(column_ids):
            mat[:,key] = self.column(id)
        return mat
    
    def blockwise_matrix(self, block_id):
        start_idx = block_id * self.d
        stop_idx = start_idx + self.d
        column_ids = list(cp.arange(start_idx, stop_idx))
        # Never generate a dense partial matrix that is larger than measurements by measurements
        if len(column_ids) > self.rows:
            print(f'WARNING: CANNOT GENERATE A MATRIX THIS LARGE; TRUNCATING columns from {len(column_ids)} to {self.rows}')
            column_ids = column_ids[0:self.rows]
        mat = cp.zeros((self.rows, len(column_ids))) 
        for key, id in enumerate(column_ids):
            mat[:,key] = self.column(id.item())
        return mat
    
    def full_matrix(self):
        m = self.rows
        n = self.cols
        mat = cp.zeros((m, n))
        if m < n:
            for j in range(n):
                mat[:,j] = self.column(j)
        else:
            assert self.transpose
            for j in range(m):
                mat[j,:] = self.column(j)
        return mat
                