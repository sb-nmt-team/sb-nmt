import numpy as np
import torch
from torch.autograd import Variable

class BatchSampler:
  def __init__(self, dataset, src_lang, tgt_lang, batch_size):
    self.train = np.array(dataset["train"])
    self.dev = np.array(dataset["dev"])
    self.test = np.array(dataset["test"])

    np.random.seed(42)
    self.train_indices = np.random.permutation(np.arange(len(self.train[0]), dtype=np.int32))

    self.src_lang = src_lang
    self.tgt_lang = tgt_lang

    self.batch_size = batch_size

  def __len__(self):
    return len(self.train[0]) // self.batch_size + 1

  def __iter__(self):
    self.position = 0
    return self

  def get_src(self):
    return self.src_lang

  def reset(self):
    self.position = 0

  def get_batch(self, x, y):
    x, x_mask = self.src_lang.convert_batch(x)
    y, y_mask = self.tgt_lang.convert_batch(y)

    x = Variable(torch.from_numpy(x.astype(np.int64))).contiguous()
    x_mask = Variable(torch.from_numpy(x_mask.astype(np.float32))).contiguous()

    y = Variable(torch.from_numpy(y.astype(np.int64))).contiguous()
    y_mask = Variable(torch.from_numpy(y_mask.astype(np.float32))).contiguous()

    return (x, x_mask), (y, y_mask)

  def __next__(self):
    if self.position >= len(self.train[0]):
      raise StopIteration()

    x = self.train[0][self.train_indices[self.position:self.position + self.batch_size]]
    y = self.train[1][self.train_indices[self.position:self.position + self.batch_size]]

    self.position += self.batch_size
    return self.get_batch(x, y)

