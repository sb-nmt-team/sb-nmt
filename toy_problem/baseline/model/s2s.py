import torch
import numpy as np

from torch import nn
from torch.autograd import Variable

from data import lang
from utils.hparams import merge_hparams
from model.encoder import EncoderRNN
from model.decoder import DecoderRNN
from model.translation_memory import TranslationMemory

class Seq2Seq(nn.Module):
  def __init__(self, source_lang, target_lang, hps, training_hps, searchengine=None):
    super(Seq2Seq, self).__init__()
    self.hps = hps
    self.training_hps = training_hps
    self.source_lang = source_lang
    self.target_lang = target_lang

    self.encoder = EncoderRNN(source_lang.input_size(), self.hps, self.training_hps)
    self.decoder = DecoderRNN(target_lang.input_size(), target_lang.input_size(), self.hps, self.training_hps)

    self.max_length = self.training_hps.max_length
    self.criterion = nn.NLLLoss(reduce=False, size_average=False)
    if hps.tm_init:
      self.translationmemory = TranslationMemory(self, hps=hps, searchengine=searchengine)
    else:
      self.translationmemory = None

  def translate(self, input_batch, mask, use_search=False):
    batch_size = input_batch.size()[0]
    encoder_outputs = self.encoder(input_batch)

    if use_search:
      assert self.translationmemory is not None, "No sample pairs for translation memory, did you want it?"
      self.translationmemory.fit(input_batch)
    hidden = None

    dec_input = Variable(torch.LongTensor([lang.BOS_TOKEN] * batch_size))

    if self.training_hps.use_cuda:
      dec_input = dec_input.cuda()

    translations = [[lang.BOS_TOKEN] for _ in range(batch_size)]
    converged = np.zeros(shape=(batch_size,))
    for i in range(self.max_length):
      if use_search:
        output, hidden, _ = self.decoder(dec_input, encoder_outputs, mask=mask, hidden=hidden,\
                                         translationmemory=self.translationmemory)
      else:
        output, hidden, _ = self.decoder(dec_input, encoder_outputs, mask=mask, hidden=hidden)
      _, output_idx = torch.max(output, -1)

      for j in range(batch_size):
        if translations[j][-1] != self.target_lang.get_eos():
          translations[j].append(output_idx[j].data[0])
        else:
          converged[j] = True
      dec_input = Variable(torch.LongTensor([tr[-1] for tr in translations]))

      if self.training_hps.use_cuda:
        dec_input = dec_input.cuda()

      if np.all(converged):
        break
    return [' '.join(map(self.target_lang.get_word, elem)) for elem in translations]

  def forward(self, input_batch, mask, output_batch, out_mask):
    encoder_outputs = self.encoder(input_batch)

    hidden = None

    loss = 0.0
    for i in range(out_mask.size()[1] - 1):
      output, hidden, _ = self.decoder(output_batch[:, i], encoder_outputs, mask=mask, hidden=hidden)
      loss += (self.criterion(output, output_batch[:, i + 1]) * out_mask[:, i + 1]).sum()

    loss /= out_mask.sum()
    return loss
  
  def get_hiddens_and_contexts(self, input_batch, mask, output_batch, out_mask):
    encoder_outputs = self.encoder(input_batch)
    B, *_ = input_batch.shape
    hidden = None

    loss = 0.0
    hiddens = Variable(torch.zeros((B, out_mask.size()[1] - 1,\
                                 self.hps.dec_layers * (self.hps.dec_bidirectional + 1) *\
                                 self.hps.dec_hidden_size)))
    contexts = Variable(torch.zeros((B, out_mask.size()[1] - 1,\
                                 self.hps.enc_layers * (self.hps.enc_bidirectional + 1) *\
                                 self.hps.enc_hidden_size)))
    
    for i in range(out_mask.size()[1] - 1):
      output, hidden, context = self.decoder(output_batch[:, i], encoder_outputs, mask=mask, hidden=hidden)
      hiddens[:, i, :] = hidden
      contexts[:, i, :] = context

    return contexts, hiddens


  @staticmethod
  def get_default_hparams():
    return merge_hparams(EncoderRNN.get_default_hparams(), DecoderRNN.get_default_hparams(),\
                         TranslationMemory.get_default_hparams())
