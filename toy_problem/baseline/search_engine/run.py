import pickle

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
