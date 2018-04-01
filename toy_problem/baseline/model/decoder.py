import torch
import torch.nn.functional as F

from torch import nn
from torch.autograd import Variable

from model.attn import Attn
from data import lang
from utils.hparams import HParams

class DecoderRNN(nn.Module):
  def __init__(self, input_size, output_size, hps, training_hps):
    super(DecoderRNN, self).__init__()
    self.hps = hps
    self.training_hps = training_hps
    self.num_directions = int(self.hps.dec_bidirectional) + 1
    self.output_size = output_size

    self.embedding = nn.Embedding(input_size, self.hps.dec_emb_size, padding_idx=lang.PAD_TOKEN)
    self.attn = Attn(self.hps)
    self.gru = nn.GRU(input_size=self.hps.dec_emb_size + \
                                 self.hps.enc_hidden_size * (int(self.hps.enc_bidirectional) + 1),
                      hidden_size=self.hps.dec_hidden_size,
                      num_layers=self.hps.dec_layers,
                      batch_first=True,
                      dropout=self.hps.dec_dropout,
                      bidirectional=self.hps.dec_bidirectional)

    self.out = nn.Linear(self.hps.dec_hidden_size * self.num_directions, output_size)
    self.retrivial_gate = nn.Linear(self.hps.enc_hidden_size * (int(self.hps.enc_bidirectional) + 1) + \
                                 self.hps.dec_hidden_size * (int(self.hps.dec_bidirectional) + 1) +\
                                 self.hps.dec_hidden_size * (int(self.hps.dec_bidirectional) + 1), 1)

  def forward(self, input, encoder_outputs, mask, hidden=None, translationmemory=None):
    """
        input: [B,]
        encoder_outputs: [B, T, HE]
        hidden: [B, layers * directions, HD]
    """
    batch_size = input.size(0)
    if hidden is None:
      hidden = self.init_hidden(batch_size)
    embedded = self.embedding(input)
    context = self.attn(hidden, encoder_outputs, mask)
    assert context.shape[0] == batch_size, len(context.shape) == 2
    if translationmemory is not None: # calculate scores q
      hidden_state_from_memory = translationmemory.match(context)

      retrivial_gate_input = torch.cat((hidden.permute(1, 0, 2).contiguous().view(batch_size, -1),\
                                        hidden_state_from_memory.permute(1, 0, 2).contiguous().view(batch_size, -1),\
                                        context.contiguous().view(batch_size, -1)), 1)
      retrivial_gate = torch.sigmoid(self.retrivial_gate(retrivial_gate_input))
    if translationmemory is not None:
      hidden = retrivial_gate * hidden + (1 - retrivial_gate) * hidden_state_from_memory
    rnn_input = torch.cat((embedded, context), -1).view(batch_size, 1, -1)

    output, next_hidden = self.gru(rnn_input, hidden)
    output = self.out(output).view(batch_size, self.output_size)
    output = F.log_softmax(output, -1)
#     if search_engine is not None:
#       output = retrivial_gate * output + (1 - retrivial_gate) * output_from_memory

    return output, next_hidden, context

  def init_hidden(self, batch_size):
    result = Variable(torch.zeros(self.hps.dec_layers * self.num_directions,
                                  batch_size,
                                  self.hps.dec_hidden_size))
    if self.training_hps.use_cuda:
      return result.cuda()
    else:
      return result

  @staticmethod
  def get_default_hparams():
    return HParams(
      dec_emb_size=128,
      dec_hidden_size=128,
      dec_dropout=0.1,
      dec_layers=1,
      dec_bidirectional=True
    )
