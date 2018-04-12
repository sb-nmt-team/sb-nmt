#!/usr/bin/env python3
# coding=utf-8
import sys
import os
import torch
import argparse
import json
import numpy as np
sys.path.append('..')

from model import s2s
from utils.translation_utils import run_translation
from utils.trainer import Trainer
from data.lang import read_problem
from search_engine import SearchEngine, OverfittedSearchEngine

# add more tests
def main():
  sys.stderr.write(sys.executable + "\n")
  os.environ['CUDA_VISIBLE_DEVICES'] = "1"
  if torch.cuda.is_available():
    sys.stderr.write("Running on device {}\n".format(torch.cuda.current_device()))
  parser = argparse.ArgumentParser()
  parser.add_argument('--params')
  parser.add_argument('--training_params')
  parser.add_argument('--model_state')
  #parser.add_argument('--src_path')
  #parser.add_argument('--tgt_path')
  parser.add_argument('--dataset')

  args = parser.parse_args()

  hps = s2s.Seq2Seq.get_default_hparams()
  if args.params:
    with open(args.params, encoding='utf-8') as fin:
      hps = hps.parse_dict(json.load(fin))

  training_params = Trainer.get_default_hparams()
  if args.training_params:
    with open(args.training_params, encoding='utf-8') as fin:
      training_params = training_params.parse_dict(json.load(fin))

  full_dataset, src, tgt = read_problem(args.dataset, n_sents=None)
  searchengine = None
  if hps.tm_init:
    if hps.tm_overfitted:
      searchengine = OverfittedSearchEngine()
    else:
      searchengine = SearchEngine()
    searchengine.load(hps.tm_bin_path)
    searchengine.set_dictionary(full_dataset)
    sys.stderr.write("Using searchengine: {}\n".format(searchengine.__class__))

  #dataset_name = "../../preprocessed/he-en/"
  #full_dataset, src, tgt = read_problem(dataset_name, n_sents=None)
    
  #src = Lang(args.src_path)
  #tgt = Lang(args.tgt_path)
  model = s2s.Seq2Seq(src, tgt, hps, training_params, searchengine)
  if training_params.use_cuda:
      model = model.cuda()

  state_dict = torch.load(args.model_state)
  model.load_state_dict(state_dict)

  sys.stderr.write("Ready!\n")
  sents = []
  for sent in sys.stdin:
      sent = sent.strip().split()
      sents.append(sent)
  
  sents = np.array(sents)
  for sent in run_translation(src, model, sents, training_params):
      print(sent)


    
if __name__ == '__main__':
  main()
