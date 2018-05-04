#!/usr/bin/env python3
# coding=utf-8
import argparse
import json
import torch
import os
import sys
import logging
import logging.config


sys.path.append('..')
from model import s2s
from utils.trainer import Trainer
from utils.launch_utils import LOGGING_BASE, find_latest_experiment, \
  create_new_experiment, get_hps, log_experiment_info, log_parameters_info, log_dataset_info, \
  logger, translate_to_all_loggers
from data.lang import read_problem
from data.batcher import BatchSampler
from search_engine import SearchEngine, OverfittedSearchEngine
from tensorboardX import SummaryWriter

TASK_NAME = "toy_problem"
TRAINED_MODELS_FOLDER = os.path.join(os.path.abspath(os.path.join(__file__ ,"../../..")), "trained_models")
DATASET_DIR = os.path.join(os.path.abspath(os.path.join(__file__ ,"../../..")), "preprocessed")


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--params')
  parser.add_argument('--training_params')
  args = parser.parse_args()

  # todo logdir
  # todo load from logdir if possible, load params
  training_params = Trainer.get_default_hparams()
  if args.training_params:
    with open(args.training_params, encoding='utf-8') as fin:
      training_params = training_params.parse_dict(json.load(fin))

  logger_config = LOGGING_BASE

  model_name = os.path.join(training_params.prefix, training_params.model_name)
  model_folder = os.path.join(TRAINED_MODELS_FOLDER, model_name)

  latest_folder = find_latest_experiment(model_folder) if os.path.exists(model_folder) else None
  latest_folder = None if training_params.force_override else latest_folder
  new_folder = create_new_experiment(model_folder, latest_folder)

  logger_config['handlers']['debug']['filename'] = os.path.join(new_folder, 'debug_logs')
  logger_config['handlers']['stdout']['filename'] = os.path.join(new_folder, 'stdout_logs')
  logging.config.dictConfig(logger_config)


  logger.info("Using python binary at {}".format(sys.executable))

  os.environ['CUDA_VISIBLE_DEVICES'] = str(training_params.cuda_visible_devices)
  if torch.cuda.is_available() and training_params.use_cuda:
    logger.info('GPU found, running on device {}'.format(torch.cuda.current_device()))
  elif training_params.use_cuda:
    logger.warning('GPU not found, running on CPU. Overriding use_cuda to False.')
    training_params.set('use_cuda', False)
  else:
    logger.debug('GPU found, but use_cuda=False, consider using GPU.')

  log_experiment_info(model_name, new_folder, latest_folder)

  hps = get_hps(new_folder, args)


  dataset_path = os.path.join(DATASET_DIR, hps.dataset)
  full_dataset, src, tgt = read_problem(dataset_path, n_sents=None)
  dataset, src, tgt = read_problem(dataset_path, n_sents=len(full_dataset["train"][0]) // hps.fraction)
  
  #DELETE ME
  #print(len(dataset["dev"]))
  #n = len(dataset["dev"]) // hps.fraction
  #print(n)
  #new_dev = (dataset["dev"][0][:n], dataset["dev"][1][:n])
  #dataset["dev"] = new_dev 
  #END DELTE ME

  training_params.set('logdir', new_folder)
  log_parameters_info(hps, training_params)
  log_dataset_info(hps, full_dataset, dataset)

  batch_sampler = BatchSampler(dataset,
                               src_lang=src,
                               tgt_lang=tgt,
                               batch_size=training_params.batch_size)
  searchengine = None
  if hps.tm_init:
    logger.info("Using translation memory.")
    if hps.tm_overfitted:
      logger.info("Using overfitted search engine.")
      searchengine = OverfittedSearchEngine()
    else:
      logger.info("Using normal search engine.")
      searchengine = SearchEngine()
    searchengine.load(hps.tm_bin_path)
    searchengine.set_dictionary(full_dataset)

  writer = SummaryWriter(log_dir=training_params.logdir)

  model = s2s.Seq2Seq(src, tgt, hps, training_params, writer=writer, searchengine=searchengine)

  with open(os.path.join(new_folder, "model.meta"), "w") as fout:
    fout.write(repr(model))
  translate_to_all_loggers(repr(model))


  trainer = Trainer(model, batch_sampler, hps, training_params, writer, searchengine)
  trainer.train()
  writer.export_scalars_to_json("./all_scalars.json")
  writer.close()

if __name__ == '__main__':
  main()
