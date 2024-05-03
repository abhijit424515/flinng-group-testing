# unverified

import numpy as np
from typing import List
import threading

class Flinng:
  """
  Flinng class represents a data structure for approximate nearest neighbor search.

  Args:
    num_rows (int): The number of rwrite basic docstring for the class belowows in the Flinng structure.
    cells_per_row (int): The number of cells per row in the Flinng structure.
    num_hashes (int): The number of hash tables in the Flinng structure.
    hash_range (int): The range of hash values in the Flinng structure.

  Attributes:
    num_rows (int): The number of rows in the Flinng structure.
    cells_per_row (int): The number of cells per row in the Flinng structure.
    num_hash_tables (int): The number of hash tables in the Flinng structure.
    hash_range (int): The range of hash values in the Flinng structure.
    inverted_flinng_index (List[List[int]]): The inverted Flinng index.
    cell_membership (List[List[int]]): The cell membership information.
    total_points_added (int): The total number of points added to the Flinng structure.

  """

  def __init__(
    self, num_rows: int, cells_per_row: int, num_hashes: int, hash_range: int
  ):
    self.num_rows = num_rows
    self.cells_per_row = cells_per_row
    self.num_hash_tables = num_hashes
    self.hash_range = hash_range
    self.inverted_flinng_index = [[] for _ in range(hash_range * num_hashes)]
    self.cell_membership = [[] for _ in range(num_rows * cells_per_row)]
    self.total_points_added = 0

  def add_points(self, hashes: List[int]):
    """
    Add points to the Flinng structure.

    Args:
      hashes (List[int]): The list of hash values.

    """

    num_points = len(hashes) // self.num_hash_tables
    random_buckets = np.random.randint(
      self.cells_per_row, size=(self.num_rows, num_points)
    )
    random_buckets += np.arange(num_points) % self.num_rows * self.cells_per_row

    def worker(start, end):
      for table in range(self.num_hash_tables):
        for point in range(start, end):
          hash_val = hashes[point * self.num_hash_tables + table]
          hash_id = table * self.hash_range + hash_val
          for row in range(self.num_rows):
            self.inverted_flinng_index[hash_id].append(
              random_buckets[row, point]
            )

      for point in range(start, end):
        for row in range(self.num_rows):
          self.cell_membership[random_buckets[row, point]].append(
            self.total_points_added + point
          )

    num_threads = 8  # Adjust as needed
    chunk_size = num_points // num_threads
    threads = []
    for i in range(num_threads):
      start = i * chunk_size
      end = start + chunk_size if i < num_threads - 1 else num_points
      thread = threading.Thread(target=worker, args=(start, end))
      thread.start()
      threads.append(thread)

    for thread in threads:
      thread.join()

    self.total_points_added += num_points
    self.prepare_for_queries()

  def query(self, hashes: List[int], top_k: int):
    """
    Perform a query on the Flinng structure.

    Args:
      hashes (List[int]): The list of hash values.
      top_k (int): The number of nearest neighbors to retrieve.

    Returns:
      List[int]: The list of indices of the nearest neighbors.

    """

    num_queries = len(hashes) // self.num_hash_tables
    results = np.zeros(top_k * num_queries, dtype=np.uint64)

    def worker(query_id, start, end):
      for idx in range(start, end):
        counts = np.zeros(self.num_rows * self.cells_per_row, dtype=np.uint32)
        for rep in range(self.num_hash_tables):
          index = (
            rep * self.hash_range + hashes[idx * self.num_hash_tables + rep]
          )
          for small_index in self.inverted_flinng_index[index]:
            counts[small_index] += 1

        sorted_indices = np.argsort(counts)[::-1]
        num_counts = np.zeros(self.total_points_added, dtype=np.uint8)
        num_found = 0

        for rep in range(self.num_hash_tables + 1):
          for bin_idx in sorted_indices:
            for point in self.cell_membership[bin_idx]:
              if num_counts[point] == self.num_rows:
                results[top_k * query_id + num_found] = point
                num_found += 1
                if num_found == top_k:
                  break
            if num_found == top_k:
              break
          if num_found == top_k:
            break

    num_threads = 8  # Adjust as needed
    chunk_size = num_queries // num_threads
    threads = []
    for i in range(num_threads):
      start = i * chunk_size
      end = start + chunk_size if i < num_threads - 1 else num_queries
      thread = threading.Thread(target=worker, args=(i, start, end))
      thread.start()
      threads.append(thread)

    for thread in threads:
      thread.join()

    return results.tolist()

  def prepare_for_queries(self):
    """
    Prepare the Flinng structure for queries.

    """

    for index in range(len(self.inverted_flinng_index)):
      self.inverted_flinng_index[index] = np.unique(
        self.inverted_flinng_index[index]
      ).tolist()
