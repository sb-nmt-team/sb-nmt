import torch
import numpy as np
import tqdm
from torch.autograd import Variable

import gc

from metrics.bleu import bleu_from_lines, vowel_bleu_from_lines
from utils.hparams import HParams
from utils.translation_utils import run_translation
# fix it

import matplotlib.pyplot as plt
import os
import time

class Trainer:
  def __init__(self, model, batch_sampler, hps, training_hps, train=True):
    self.timestamp = time.strftime("%Y-%m-%d_%H_%M_%S", time.gmtime())
    self.result_dir = os.path.join(training_hps.result_dir, training_hps.model_name + "_on_" + self.timestamp)
    # self.save_path = os.path.join(self.result_dir,)
    os.makedirs(self.result_dir)

    self.hps = hps
    self.training_hps = training_hps

    self.model = model
    self.batch_sampler = batch_sampler
    self.losses = []

    self.metrics = {
      "bleu": [],
      "vowel-bleu": []
    }

    with open(os.path.join(self.result_dir, "trainer.meta"), "w") as f:
      f.write(repr(self.model) + "\n")
      f.write("\t==========================\n")
      f.write(self.timestamp + " GMT\n")

  def reset(self):
    pass

  def validate(self):
    test_data = self.batch_sampler.dev[0]
    translation = run_translation(self.batch_sampler.get_src(), self.model, test_data,
                                       self.training_hps)
    real_translation = [' '.join(x) for x in self.batch_sampler.dev[1]]
    return {
      "bleu": bleu_from_lines(real_translation, translation),
      "vowel-bleu": vowel_bleu_from_lines(real_translation, translation)
    }

  def update_metrics(self, update):
    for metric in self.metrics:
      if metric in update:
        self.metrics[metric].append(update[metric])

  def print_metrics(self):
    for metric in self.metrics:
      print("{0}: {1}".format(metric, self.metrics[metric][-1]))

  def train(self):
    # plt.ion()
    # plt.show()

    self.model.train()

    # todo multiple optimizers
    optimizer = torch.optim.Adam(self.model.parameters(), lr=self.training_hps.starting_learning_rate)

    if self.training_hps.use_cuda:
      self.model = self.model.cuda()
      # optimizer = optimizer.cuda()

    for epoch_id in range(self.training_hps.n_epochs):

      for batch_id, ((input, input_mask), (output, output_mask)) in \
        tqdm.tqdm(enumerate(self.batch_sampler), total=len(self.batch_sampler)):
        if self.training_hps.use_cuda:
          input = input.cuda()
          input_mask = input_mask.cuda()
          output = output.cuda()
          output_mask = output_mask.cuda()

        loss = self.model(input, input_mask, output, output_mask)
        optimizer.zero_grad()

        loss.backward()
        torch.nn.utils.clip_grad_norm(self.model.parameters(), self.training_hps.clip)
        optimizer.step()

        # todo create hooks
        # it's really doubtfull to hold all of them
        if self.training_hps.use_cuda:
          self.losses.append(loss.cpu().data[0])
        else:
          self.losses.append(loss.data[0])
        # print(loss.cpu().data[0])
        if (batch_id * self.batch_sampler.batch_size) % 1000 == 0:
          # display.clear_output(wait=True)
          print("Last 10 loses mean", np.mean(self.losses[-10:]))
          # plt.plot(self.losses)
          # plt.show(block=False)
          # plt.draw()

      gc.collect()
      if self.training_hps.use_cuda:
        torch.cuda.empty_cache()

      self.model.eval()
      self.update_metrics(self.validate())

      print("After epoch ", epoch_id)
      self.print_metrics()

      self.model.train()

      # todo redo the saving
      torch.save(self.model.state_dict(), os.path.join(self.result_dir, "last_state.ckpt"))
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
      result_dir="./",
      model_name="Model",
    )