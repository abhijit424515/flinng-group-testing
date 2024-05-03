import numpy as np
from joblib import Parallel, delayed
from helpers.mmh import murmurhash3_x86_32
from cls.lsh import LSH


class FLINNG:
    def __init__(
        self,
        row_count: np.uint32,
        blooms_per_row: np.uint32,
        hashes: np.ndarray,
        num_hashes_generated: np.uint32,
        hash_function: LSH,
        hash_bits: np.uint32,
        hash_repeats: np.uint32,
        num_points: np.uint32,
    ):
        self.row_count = row_count
        self.blooms_per_row = blooms_per_row
        self.num_bins = blooms_per_row * row_count
        self.hash_repeats = hash_repeats
        self.num_points = num_points
        self.internal_hash_bits = hash_bits
        self.internal_hash_length = 1 << hash_bits
        self.rambo_array = np.empty(
            (hash_repeats * self.internal_hash_length,), dtype=object
        )
        self.meta_rambo = np.empty((row_count * blooms_per_row,), dtype=object)
        self.hashes = hashes
        self.num_hashes_generated = num_hashes_generated
        self.hash_function = hash_function

        print("Getting row indices")
        row_indices_arr = Parallel(n_jobs=-1)(
            delayed(lambda i: self.get_hashed_row_indices(i))(i)
            for i in range(num_points)
        )
        row_indices_arr = np.array(row_indices_arr)

        print("Creating meta rambo")
        for point in range(num_points):
            hashvals = row_indices_arr[point]
            for r in range(row_count):
                self.meta_rambo[hashvals[r] + blooms_per_row * r] = []

        for point in range(num_points):
            hashvals = row_indices_arr[point]
            for r in range(row_count):
                self.meta_rambo[hashvals[r] + blooms_per_row * r].append(point)

        print("Sorting meta rambo")
        self.meta_rambo = Parallel(n_jobs=-1)(
            delayed(lambda arr: sorted(arr))(lst) for lst in self.meta_rambo
        )
        self.meta_rambo = np.array(self.meta_rambo)

        self.do_inserts(row_indices_arr)
        del row_indices_arr

    def get_hashed_row_indices(self, index: np.uint32) -> np.ndarray:
        key = str(index)
        hashvals = np.zeros(self.row_count, dtype=np.uint32)
        for i in range(self.row_count):
            hashvals[i] = murmurhash3_x86_32(key.encode(), i) % self.blooms_per_row
        return hashvals

    def do_inserts(self, row_indices_arr: np.ndarray):
        print("Populating FLINNG")

        def chunk(rep):
            for index in range(self.num_points):
                row_indices = row_indices_arr[index]
                for r in range(self.row_count):
                    b = row_indices[r]
                    self.rambo_array[
                        rep * self.internal_hash_length
                        + self.hashes[rep * self.num_points + index]
                    ].append(r * self.blooms_per_row + b)

        Parallel(n_jobs=-1)(delayed(chunk)(rep) for rep in range(self.hash_repeats))

    def finalize_construction(self):
        print("Sorting FLINNG")

        def chunk(arr: np.ndarray):
            arr.sort()
            return list(dict.fromkeys(arr))

        self.rambo_array = Parallel(n_jobs=-1)(
            delayed(chunk)(self.rambo_array[i])
            for i in range(self.internal_hash_length * self.hash_repeats)
        )

    def query(self, data_ids, data_vals, data_marker, query_goal, query_output):
        hashes, indices = self.hash_function.getHash(
            data_ids, data_vals, data_marker, 1, 1, 1, 0
        )

        counts = np.zeros(self.num_bins, dtype=int)
        for rep in range(self.hash_repeats):
            index = self.internal_hash_length * rep + hashes[rep]
            size = len(self.rambo_array[index])
            for small_index in range(size):
                counts[self.rambo_array[index][small_index]] += 1

        sorted_bins = [[] for _ in range(self.hash_repeats + 1)]
        for i in range(self.num_bins):
            sorted_bins[counts[i]].append(i)

        if self.row_count > 2 or self.num_points < 4000000:
            num_counts = np.zeros(self.num_points, dtype=int)
            num_found = 0
            for rep in range(self.hash_repeats, -1, -1):
                for bin_ in sorted_bins[rep]:
                    for point in self.meta_rambo[bin_]:
                        num_counts[point] += 1
                        if num_counts[point] == self.row_count:
                            query_output[num_found] = point
                            num_found += 1
                            if num_found == query_goal:
                                return
        else:
            num_counts = np.zeros(self.num_points // 8 + 1, dtype=np.uint8)
            num_found = 0
            for rep in range(self.hash_repeats, -1, -1):
                for bin_ in sorted_bins[rep]:
                    for point in self.meta_rambo[bin_]:
                        if num_counts[point // 8] & (1 << (point % 8)):
                            query_output[num_found] = point
                            num_found += 1
                            if num_found == query_goal:
                                return
                        else:
                            num_counts[point // 8] |= 1 << (point % 8)
