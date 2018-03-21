import editdistance
import heapq
import numpy as np
import pickle
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
  return 1 - editdistance.eval(first, second) / max(len(first), len(second))

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

def dump_bin(results, outfile):
  with open(outfile, "wb") as file:
    pickle.dump(results, file)

def dump_text(results, outfile):
  dataset = results["dataset"]
  result = results["result"]

  with open(outfile, "w") as  file:
    for sent, neighbours in result.items():
      file.write(
          "%s: %s\n" % (
              sent,
              ' '.join(map(lambda x: "[%.3f, %s]" % (x[0], dataset[x[1]]), 
                           neighbours))
          )
      )

if __name__ == '__main__':
  TRAIN_SET_PATH = '../preprocessed/he-en/src.train.txt'
  DATASET_PATHS = ['../preprocessed/he-en/src.train.txt',
                   '../preprocessed/he-en/src.test.txt',
                   '../preprocessed/he-en/src.dev.txt']

  N_NEIGHBOURS = 100
  N_JOBS = 16
  N_CHUNKS = 1600

  train_set = read_dataset(TRAIN_SET_PATH)
  train_set = list(set(train_set))
  
  query_set = read_queries(DATASET_PATHS)
  query_set = list(set(query_set))

  print("DB size:", len(train_set))
  print("Queries size:", len(query_set))

  search_engine = find_closest_train_sentences(
    query_set, train_set, N_NEIGHBOURS, N_JOBS, N_CHUNKS, True)

  dump_bin(search_engine, "./se.bin")
  # dump_text(search_engine, "./se.txt")
