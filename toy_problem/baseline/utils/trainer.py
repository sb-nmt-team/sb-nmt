import torch
import numpy as np
import tqdm
from torch.autograd import Variable

import gc

from metrics.bleu import bleu_from_lines, vowel_bleu_from_lines
from utils.hparams import HParams
from utils.translation_utils import run_translation
from utils.launch_utils import log_func, translate_to_all_loggers
# fix it

#import matplotlib.pyplot as plt
import os
import time
import itertools

class Trainer:
  @log_func
  def __init__(self, model, batch_sampler, hps, training_hps, train=True):
    self.hps = hps
    self.training_hps = training_hps

    self.model = model
    self.batch_sampler = batch_sampler
    self.losses = []

    self.metrics = {
      "bleu": [],
      "vowel-bleu": []
    }

  def reset(self):
    pass

  @log_func
  def validate(self):
    self.model.eval()
    test_data = self.batch_sampler.dev[0]
    translation = run_translation(self.batch_sampler.get_src(), self.model, test_data,
                                       self.training_hps)
    real_translation = [' '.join(x) for x in self.batch_sampler.dev[1]]
    self.model.train()
    return {
      "bleu": bleu_from_lines(real_translation, translation),
      "vowel-bleu": vowel_bleu_from_lines(real_translation, translation)
    }

  @log_func
  def update_metrics(self, update):
    for metric in self.metrics:
      if metric in update:
        self.metrics[metric].append(update[metric])

  def print_metrics(self):
    for metric in self.metrics:
      print("{0}: {1}".format(metric, self.metrics[metric][-1]))

  @log_func
  def train(self):
    # plt.ion()
    # plt.show()

    # todo it should load
    self.model.train()


    # todo multiple optimizers
    optimizer = torch.optim.Adam(self.model.parameters(), lr=self.training_hps.starting_learning_rate)

    if self.training_hps.use_cuda:
      self.model = self.model.cuda()
      # optimizer = optimizer.cuda()
    self.training_hps.use_tm_on_test = False
    for epoch_id in range(self.training_hps.n_epochs):

      for batch_id, ((input, input_mask), (output, output_mask)) in \
        tqdm.tqdm(enumerate(self.batch_sampler), total=len(self.batch_sampler)):
        self.model.train()
        if self.training_hps.use_cuda:
          input = input.cuda()
          input_mask = input_mask.cuda()
          output = output.cuda()
          output_mask = output_mask.cuda()

        loss = self.model(input, input_mask, output, output_mask, use_search=False)
        optimizer.zero_grad()

        loss.backward()
        torch.nn.utils.clip_grad_norm(self.model.parameters(), self.training_hps.clip)
        optimizer.step()
        gc.collect()
        if self.training_hps.use_cuda:
            torch.cuda.empty_cache()

        # it's really doubtfull to hold all of them
        if self.training_hps.use_cuda:
          self.losses.append(loss.cpu().data[0])
        else:
          self.losses.append(loss.data[0])

        if (batch_id * self.batch_sampler.batch_size) % 1000 == 0:
          translate_to_all_loggers("Last 10 loses mean {0:4.3f}".format(np.mean(self.losses[-10:])))

      gc.collect()
      if self.training_hps.use_cuda:
        torch.cuda.empty_cache()

      self.update_metrics(self.validate())

      translate_to_all_loggers("Epoch ended, after epoch metrics: {}".format(epoch_id))
      self.print_metrics()


      # todo redo the saving
      torch.save(self.model.state_dict(), os.path.join(self.training_hps.logdir, "last_state.ckpt"))
      gc.collect()
      if self.training_hps.use_cuda:
        torch.cuda.empty_cache()
    self.training_hps.use_tm_on_test = True
    if not self.hps.tm_init:
      return
    translate_to_all_loggers("Starting the trainer with search database.")
    optimizer = torch.optim.Adam(itertools.chain.from_iterable((self.model.parameters(), self.model.translationmemory.parameters())), lr=self.training_hps.starting_learning_rate)
    # todo copypaste
    for epoch_id in range(self.training_hps.n_tm_epochs):
      for batch_id, ((input, input_mask), (output, output_mask)) in \
        tqdm.tqdm(enumerate(self.batch_sampler), total=len(self.batch_sampler)):
        if self.training_hps.use_cuda:
          input = input.cuda()
          input_mask = input_mask.cuda()
          output = output.cuda()
          output_mask = output_mask.cuda()

        loss = self.model(input, input_mask, output, output_mask, use_search=True)
        optimizer.zero_grad()

        loss.backward()
        torch.nn.utils.clip_grad_norm(self.model.parameters(), self.training_hps.clip)
        optimizer.step()
        gc.collect()
        if self.training_hps.use_cuda:
            torch.cuda.empty_cache()


        if self.training_hps.use_cuda:
          self.losses.append(loss.cpu().data[0])
        else:
          self.losses.append(loss.data[0])
        # print(loss.cpu().data[0])
        if (batch_id * self.batch_sampler.batch_size) % 1000 == 0:
          translate_to_all_loggers("Last 10 loses mean {0:4.3f}".format(np.mean(self.losses[-10:])))

      gc.collect()
      if self.training_hps.use_cuda:
        torch.cuda.empty_cache()

      self.update_metrics(self.validate())

      translate_to_all_loggers("Epoch ended, after epoch metrics: {}".format(epoch_id))
      self.print_metrics()

      # todo redo the saving
      torch.save(self.model.state_dict(), os.path.join(self.training_hps.logdir, "last_state.ckpt"))
      gc.collect()
      if self.training_hps.use_cuda:
        torch.cuda.empty_cache()
  def get_metrics(self):
    # todo return metrics and not this stuff
    return self.losses, self.bleu

  @staticmethod
  def get_default_hparams():
    return HParams(
      use_cuda=False,
      max_length=15,
      batch_size=128,
      n_epochs=40,
      clip=0.25,
      starting_learning_rate=1e-3, # todo
      learning_rate_strategy="constant_decay", # todo
      optimizer="Adam", # todo,
      prefix="",
      model_name="",
      logdir="",
      use_tm_on_test=False,
      n_tm_epochs=10,
      cuda_visible_devices=1,
    )
