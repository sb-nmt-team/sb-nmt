import editdistance
import heapq
import numpy as np
import sys
import multiprocessing
import tqdm

def read_dataset(dataset_path):
  dataset = []
  with open(dataset_path) as file:
    dataset = [tuple(line.split()) for line in file]
  return dataset

def read_queries(queries_paths):
  data = []
  for path in queries_paths:
      data += read_dataset(path)
  return data

def calucate_metric(first, second):
  return 1 - 1.0 * editdistance.eval(first, second) / \
             max(len(first), len(second))

def find_closest(sentence, train_set, n_neighbours):
  heap = []
  for index, candidate in enumerate(train_set):
    if sentence == candidate:
      continue
    metric = calucate_metric(sentence, candidate)
    if len(heap) < n_neighbours:
      heapq.heappush(heap, (metric, index))
    elif heap[0][0] < metric:
      heapq.heappushpop(heap, (metric, index))

  return sorted(heap, reverse=True)

def find_closest_batch(batch, train_set, n_neighbours):
  result = {}
  for sentence in batch:
    result[sentence] = find_closest(sentence, train_set, n_neighbours)
  return result

def find_closest_batch_wrapper(args):
  return find_closest_batch(*args)

def find_closest_train_sentences(
    sentences, train_set, n_neighbours, n_jobs, n_chunks=None, verbose=False):
  if n_chunks is None:
    n_chunks = n_jobs

  result = {}
  queries = np.array_split(sentences, n_chunks)
  args = map(lambda x: (list(x), train_set, n_neighbours), queries)
  
  pool = multiprocessing.Pool(n_jobs)
  iterable = pool.imap_unordered(find_closest_batch_wrapper, args)
  if verbose:
    iterable = tqdm.tqdm(iterable, total=n_chunks)
  
  for batch_result in iterable:
    result.update(batch_result)

  return {"result": result, "dataset": train_set}
