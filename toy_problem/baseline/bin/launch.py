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

def main():
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

  dataset, src, tgt = read_problem(dataset_name, n_sents=None)
  dataset_size = len(dataset["train"][0])
  n_sents = dataset_size // fraction
  # n_sents = 100

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

  model = s2s.Seq2Seq(src, tgt, hps, training_params)

  trainer = Trainer(model, batch_sampler, hps, training_params)
  trainer.train()

if __name__ == '__main__':
  main()