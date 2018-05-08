import torch
import numpy as np
import tqdm
from torch.autograd import Variable

import gc
import time
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
  def __init__(self, model, batch_sampler, hps, training_hps, writer, train=True):
    self.hps = hps
    self.training_hps = training_hps

    self.model = model
    self.batch_sampler = batch_sampler
    self.losses = []

    self.metrics = {
      "bleu": [],
      "vowel-bleu": [],
      "bleu-train": [],
      "vowel-bleu-train": []
    }
    self.writer = writer

  def reset(self):
    pass

  @log_func
  def validate(self):
    self.model.eval()
    #test_data = self.batch_sampler.dev[0][:100]
    test_data = self.batch_sampler.dev[0]
    translate_to_all_loggers("Validating on {}".format(len(test_data)))
    translation = run_translation(self.batch_sampler.get_src(), self.model, test_data,
                                       self.training_hps)
    real_translation = [' '.join(x) for x in self.batch_sampler.dev[1]]

    translate_to_all_loggers("On validation got sentences (src -> translated (correct)):" )
    for i in range(30):
        translate_to_all_loggers("{} -> {} ({})".format(test_data[i], translation[i], real_translation[i]))
    

    translate_to_all_loggers("Validated on {}, {}".format(len(real_translation), len(translation)))
    result = {
      "bleu": bleu_from_lines(real_translation, translation),
      "vowel-bleu": vowel_bleu_from_lines(real_translation, translation)
    }
    
    n_train = len(test_data)
    test_data = self.batch_sampler.train[0][:n_train]
    real_translation = [' '.join(x) for x in self.batch_sampler.train[1][:n_train]]
    translation = run_translation(self.batch_sampler.get_src(), self.model, test_data,
                                       self.training_hps)

    translate_to_all_loggers("On train got sentences (src -> translated (correct)):" )
    for i in range(30):
        translate_to_all_loggers("{} -> {} ({})".format(test_data[i], translation[i], real_translation[i]))

    translate_to_all_loggers("Validated on {}, {}".format(len(real_translation), len(translation)))

    result["bleu-train"] = bleu_from_lines(real_translation, translation)
    result["vowel-bleu-train"] = vowel_bleu_from_lines(real_translation, translation)

    self.model.train()
    return result

  @log_func
  def update_metrics(self, update, step, prefix):
    #is_testing = not self.model.train
    for metric_name in self.metrics:
      if metric_name not in update:
        translate_to_all_loggers("{} not found in metrics".format(metric_name))

      current_metric_value = update[metric_name]
      if prefix == 'tm':
        for pr in ['normal', 'tm']:
          self.writer.add_scalar('validation/{}_{}'.format(pr, metric_name), current_metric_value,
                                step)
      else:
        self.writer.add_scalar('validation/{}_{}'.format(prefix, metric_name), current_metric_value,
                               step)
      self.metrics[metric_name].append(current_metric_value)

  def print_metrics(self):
    for metric in self.metrics:
      translate_to_all_loggers("{0}: {1:4.3f}".format(metric, self.metrics[metric][-1]))

  def train_loop(self, optimizer, begin_epoch, end_epoch, prefix="normal", set_model_to_train=True, use_search=False):
    for epoch_id in range(begin_epoch, end_epoch):
      for batch_id, ((input, input_mask), (output, output_mask)) in \
        tqdm.tqdm(enumerate(self.batch_sampler), total=len(self.batch_sampler)):

        if set_model_to_train:
          self.model.train()
        else:
          self.model.eval()

        if self.training_hps.use_cuda:
          input = input.cuda()
          input_mask = input_mask.cuda()
          output = output.cuda()
          output_mask = output_mask.cuda()

        loss = self.model(input, input_mask, output, output_mask, use_search=use_search)
        optimizer.zero_grad()

        loss.backward()
        if use_search:
          for param_name, value in itertools.chain.from_iterable((self.model.translationmemory.named_parameters(),self.model.named_parameters())):
              if hasattr(value, "grad") and value.grad is not None:
                self.writer.add_scalar('grads_search/{}'.format(param_name),
                                value.grad.cpu().mean(), epoch_id * len(self.batch_sampler) + batch_id)

          torch.nn.utils.clip_grad_norm(itertools.chain.from_iterable((self.model.translationmemory.parameters(),self.model.parameters())), self.training_hps.clip)
        else:
          for param_name, value in self.model.named_parameters():
              if hasattr(value, "grad") and value.grad is not None:
                self.writer.add_scalar('grads_no_search/{}'.format(param_name),
                                value.grad.cpu().mean(), epoch_id * len(self.batch_sampler) + batch_id)
          torch.nn.utils.clip_grad_norm(self.model.parameters(), self.training_hps.clip)
        optimizer.step()
        gc.collect()
        if self.training_hps.use_cuda:
            torch.cuda.empty_cache()

        loss_data = loss.cpu().data[0] if self.training_hps.use_cuda else loss.data[0]
        self.losses.append(loss_data)
        self.writer.add_scalar('train/{}_loss'.format(prefix), loss_data, epoch_id * len(self.batch_sampler) + batch_id)
        for name, param in self.model.named_parameters():
            try:
                self.writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch_id * len(self.batch_sampler) + batch_id)
            except Exception as e:
                print("Failed on ", name)
                print(e)
                raise 
        for name, param in self.model.translationmemory.named_parameters():
          self.writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch_id * len(self.batch_sampler) + batch_id)

        if (batch_id * self.batch_sampler.batch_size) % 500 == 0:
            translate_to_all_loggers("Last 10 loses mean {0:4.3f}".format(np.mean(self.losses[-10:])))
           # gc.collect()
           # if self.training_hps.use_cuda:
           #     torch.cuda.empty_cache()

          #  self.update_metrics(self.validate(), epoch_id * len(self.batch_sampler), prefix)

          #  translate_to_all_loggers("Epoch {}. Validation:".format(epoch_id))
          #  self.print_metrics()

      gc.collect()
      if self.training_hps.use_cuda:
        torch.cuda.empty_cache()

      self.update_metrics(self.validate(), epoch_id * len(self.batch_sampler), prefix)

      translate_to_all_loggers("Epoch {}. Validation:".format(epoch_id))
      self.print_metrics()

      torch.save(self.model.state_dict(), os.path.join(self.training_hps.logdir, "last_state.ckpt"))
      gc.collect()
      if self.training_hps.use_cuda:
        torch.cuda.empty_cache()

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

    self.training_hps.use_tm_on_test = False
    self.train_loop(optimizer, begin_epoch=0, end_epoch=self.training_hps.n_epochs)

    translate_to_all_loggers("Starting the trainer with search database.")

    self.training_hps.use_tm_on_test = True
    if not self.hps.tm_init:
      return
    optimizer = torch.optim.Adam(itertools.chain.from_iterable((self.model.translationmemory.parameters(),self.model.parameters())),
                                 lr=self.training_hps.starting_learning_rate)
    self.train_loop(optimizer, begin_epoch=self.training_hps.n_epochs,
                    end_epoch=self.training_hps.n_tm_epochs + self.training_hps.n_epochs, prefix="tm", set_model_to_train=True, use_search=True)


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
      force_override=False
    )
