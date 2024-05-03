# unverified

import numpy as np
import flinng
import helpers.old_lsh as old_lsh


def checkValidAndGetNumPoints(points, data_dimension):
    # Get the buffer information of the numpy array
    points_buf = points.request()

    # Check if the array is 2-dimensional
    if points_buf.ndim != 2:
        raise ValueError(
            "The input points must be a 2 dimensional Numpy array where each row is a single point."
        )

    # Extract the number of points and the dimension of each point
    num_points = points_buf.shape[0]
    point_dimension = points_buf.shape[1]

    # Check if the dimensions of the points are valid
    if (data_dimension != 0 and point_dimension != data_dimension) or num_points == 0:
        raise ValueError(
            f"The rows (each point) must be of dimension {data_dimension}, and there must be at least 1 row."
        )

    return num_points


class DenseFlinng32:
    def __init__(
        self, num_rows, cells_per_row, data_dimension, num_hash_tables, hashes_per_table
    ):
        self.internal_flinng = flinng.Flinng(
            num_rows, cells_per_row, num_hash_tables, 1 << hashes_per_table
        )
        self.num_hash_tables = num_hash_tables
        self.hashes_per_table = hashes_per_table
        self.data_dimension = data_dimension
        self.rand_bits = np.random.choice(
            [-1, 1], size=(num_hash_tables * hashes_per_table * data_dimension)
        )

    def addPoints(self, points):
        checkValidAndGetNumPoints(points, self.data_dimension)
        hashes = self.getHashes(points)
        self.internal_flinng.addPoints(hashes)

    def prepareForQueries(self):
        self.internal_flinng.prepareForQueries()

    def query(self, queries, top_k):
        num_queries = checkValidAndGetNumPoints(queries, self.data_dimension)
        hashes = self.getHashes(queries)
        results = self.internal_flinng.query(hashes, top_k)
        return np.array(results).reshape((num_queries, top_k))

    def getHashes(self, points):
        num_points = points.shape[0]
        return flinng.parallel_srp(
            points.flatten(),
            num_points,
            self.data_dimension,
            self.rand_bits,
            self.num_hash_tables,
            self.hashes_per_table,
        )


class SparseFlinng32:
    def __init__(
        self, num_rows, cells_per_row, num_hash_tables, hashes_per_table, hash_range_pow
    ):
        self.internal_flinng = flinng.Flinng(
            num_rows, cells_per_row, num_hash_tables, 1 << hash_range_pow
        )
        self.num_hash_tables = num_hash_tables
        self.hashes_per_table = hashes_per_table
        self.hash_range_pow = hash_range_pow
        self.seed = np.random.randint(0, 2**32)

    def addPointsSameDim(self, points):
        self.checkValidAndGetNumPoints(points, 0)
        hashes = self.getHashes(points)
        self.internal_flinng.addPoints(hashes)

    def addPoints(self, data):
        hashes = self.getHashes(data)
        self.internal_flinng.addPoints(hashes)

    def hashPoints(self, data):
        return self.getHashes(data)

    def prepareForQueries(self):
        self.internal_flinng.prepareForQueries()

    def query(self, queries, top_k):
        hashes = self.getHashes(queries)
        results = self.internal_flinng.query(hashes, top_k)
        return np.array(results).reshape((len(queries), top_k))

    def querySameDim(self, queries, top_k):
        num_queries = self.checkValidAndGetNumPoints(queries, 0)
        hashes = self.getHashes(queries)
        results = self.internal_flinng.query(hashes, top_k)
        return np.array(results).reshape((num_queries, top_k))

    def getHashes(self, data):
        if isinstance(data, np.ndarray):
            num_points = data.shape[0]
            point_dimension = data.shape[1]
            return old_lsh.parallel_densified_minhash(
                data.flatten(),
                num_points,
                point_dimension,
                self.num_hash_tables,
                self.hashes_per_table,
                self.hash_range_pow,
                self.seed,
            )
        elif isinstance(data, list):
            return old_lsh.parallel_densified_minhash(
                data,
                self.num_hash_tables,
                self.hashes_per_table,
                self.hash_range_pow,
                self.seed,
            )
