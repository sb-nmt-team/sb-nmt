#!/usr/bin/env python3
# coding=utf-8
import argparse
import json
import sys
sys.path.append('..')

from model import s2s
from utils.trainer import Trainer
from data.lang import read_problem
from data.batcher import BatchSampler
from search_engine import SearchEngine, OverfittedSearchEngine
import torch
import os


def main():
  print(sys.executable)
  os.environ['CUDA_VISIBLE_DEVICES'] = "1"
  if torch.cuda.is_available():
    print("Running on device ", torch.cuda.current_device())
  parser = argparse.ArgumentParser()
  parser.add_argument('--params')
  parser.add_argument('--training_params')
  args = parser.parse_args()

  # todo logdir
  # todo load from logdir if possible, load params
  hps = s2s.Seq2Seq.get_default_hparams()
  if args.params:
    with open(args.params, encoding='utf-8') as fin:
      hps = hps.parse_dict(json.load(fin))

  training_params = Trainer.get_default_hparams()
  if args.training_params:
    with open(args.training_params, encoding='utf-8') as fin:
      training_params = training_params.parse_dict(json.load(fin))

  # that really should be dynamic
  dataset_name = "../../preprocessed/he-en/"
  logging_dir = "../../trained_models"
  experiment_name = "experiment"
  fraction = 8
  full_dataset, src, tgt = read_problem(dataset_name, n_sents=None)
  dataset_size = len(full_dataset["train"][0])
  n_sents = dataset_size // fraction
  #n_sents = 256

  dataset, src, tgt = read_problem(dataset_name, n_sents=n_sents)



  print("model params\n", json.dumps(dict(hps.items())))
  print("training_params\n", json.dumps(dict(training_params.items())))
  print("Using dataset ", dataset_name)
  print("dataset size", dataset_size)
  print("Using {} of it".format(n_sents))
  print("Fraction ", fraction)
  


  # dummy_dataset = {
  #   "train": (["'a 'a d y r", "'a 'a h b ckh"], ["a a d i r", "e a h a v k h a"]),
  #   "test": (["'a 'a d y r", "'a 'a h b ckh"], ["a a d i r", "e a h a v k h a"]),
  #   "dev": (["'a 'a d y r", "'a 'a h b ckh"], ["a a d i r", "e a h a v k h a"])
  # }
  # print(dataset["train"][0])
  batch_sampler = BatchSampler(dataset, src, tgt, training_params.batch_size)

  searchengine = None
  if hps.tm_init:
    if hps.tm_overfitted:
      searchengine = OverfittedSearchEngine()
    else:
      searchengine = SearchEngine()
    searchengine.load(hps.tm_bin_path)
    searchengine.set_dictionary(full_dataset)
    print("Using searchengine: {}".format(searchengine.__class__))

  model = s2s.Seq2Seq(src, tgt, hps, training_params, searchengine)

  trainer = Trainer(model, batch_sampler, hps, training_params)
  trainer.train()

if __name__ == '__main__':
  main()
