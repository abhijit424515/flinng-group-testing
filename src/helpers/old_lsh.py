# unverified

import numpy as np
import threading


def combine(item1: np.uint64, item2: np.uint64) -> np.uint64:
    return item1 * np.uint64(0xC4DD05BF) + item2 * np.uint64(0x6C8702C9)


def single_densified_minhash(
    result: np.ndarray,
    point: np.ndarray,
    num_tables: int,
    hashes_per_table: int,
    hash_range_pow: int,
    random_seed: int,
) -> None:

    num_hashes_to_generate = num_tables * hashes_per_table
    prelim_result = np.full(
        num_hashes_to_generate, np.uint64(18446744073709551615)
    )  # UINT64_MAX
    binsize = np.ceil(np.uint64(18446744073709551615) / prelim_result.size)

    for i in range(point.size):
        val = point[i]
        val *= random_seed
        val ^= val >> 13
        val *= np.uint64(0x192AF017AAFFF017)
        val *= val
        hash_val = val
        binid = min(int(np.floor(val / binsize)), num_hashes_to_generate - 1)
        if prelim_result[binid] > hash_val:
            prelim_result[binid] = hash_val

    # Densify
    for i in range(num_hashes_to_generate):
        next_val = prelim_result[i]
        if next_val != np.uint64(18446744073709551615):  # UINT64_MAX
            continue
        count = 0
        while next_val == np.uint64(18446744073709551615):  # UINT64_MAX
            count += 1
            index = combine(i, count) % num_hashes_to_generate
            next_val = prelim_result[index]
            if count > 100:  # Densification failure
                next_val = np.uint64(0)
                break
        prelim_result[i] = next_val

    # Combine each K
    for table in range(num_tables):
        result[table] = prelim_result[hashes_per_table * table]
        for hash_val in range(1, hashes_per_table):
            result[table] = combine(
                prelim_result[hashes_per_table * table], result[table]
            )
        result[table] >>= 64 - hash_range_pow


def parallel_densified_minhash(
    points: np.ndarray,
    num_tables: int,
    hashes_per_table: int,
    hash_range_pow: int,
    random_seed: int,
) -> np.ndarray:

    num_points, point_dimension = points.shape
    result = np.zeros(num_tables * num_points, dtype=np.uint64)

    def compute_minhash(point_id):
        single_densified_minhash(
            points[point_id],
            result[point_id * num_tables : (point_id + 1) * num_tables],
            point_dimension,  # Adding point_dimension parameter
            num_tables,
            hashes_per_table,
            hash_range_pow,
            random_seed,
        )

    threads = []
    for point_id in range(num_points):
        thread = threading.Thread(target=compute_minhash, args=(point_id,))
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()

    return result


def parallel_srp(
    dense_data, num_points, data_dimension, random_bits, num_tables, hashes_per_table
):
    result = np.zeros(num_tables * num_points, dtype=np.uint64)

    def compute_srp(data_id):
        for rep in range(num_tables):
            hash_val = 0
            for bit in range(hashes_per_table):
                sum_val = 0
                for j in range(data_dimension):
                    val = dense_data[data_dimension * data_id + j]
                    sign = (
                        random_bits[
                            rep * hashes_per_table * data_dimension
                            + bit * data_dimension
                            + j
                        ]
                        > 0
                    )
                    sum_val += val if sign else -val
                hash_val += (sum_val > 0) << bit
            result[data_id * num_tables + rep] = hash_val

    threads = []
    for data_id in range(num_points):
        thread = threading.Thread(target=compute_srp, args=(data_id,))
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()

    return result
