import torch
import torch.nn.functional as F

from torch import nn
from torch.autograd import Variable

from model.attn import Attn
from data import lang
from utils.hparams import HParams
from utils.debug_utils import assert_shape_equal


class DecoderRNN(nn.Module):
  def __init__(self, input_size, output_size, hps, training_hps, writer=None):
    super(DecoderRNN, self).__init__()
    self.writer = writer
    self.hps = hps
    self.i = 0
    self.training_hps = training_hps
    self.num_directions = int(self.hps.dec_bidirectional) + 1
    self.output_size = output_size
    self.embedding = nn.Embedding(input_size, self.hps.dec_emb_size, padding_idx=lang.PAD_TOKEN)
    self.attn = Attn(self.hps, self.writer)
    self.gru = nn.GRU(input_size=self.hps.dec_emb_size + \
                                 self.hps.enc_hidden_size * (int(self.hps.enc_bidirectional) + 1),
                      hidden_size=self.hps.dec_hidden_size,
                      num_layers=self.hps.dec_layers,
                      batch_first=True,
                      dropout=self.hps.dec_dropout,
                      bidirectional=self.hps.dec_bidirectional)

    self.out = nn.Linear(self.hps.dec_hidden_size * self.num_directions, output_size)

  def forward(self, input, encoder_outputs, mask, hidden=None, translation_memory=None):
    """
        input: [B,]
        encoder_outputs:  [B, T, DE * HE]
        hidden: [LD * DD, B, HD]
        output: [B, trg_lang_size]
    """
    batch_size, T, _ = encoder_outputs.size()
    first_iter = False
    if hidden is None:
      first_iter = True
      hidden = self.init_hidden(batch_size)
    if __debug__:
      assert input.size() == torch.Size([batch_size])
      assert_shape_equal(hidden.size(), torch.Size([self.hps.dec_layers * (int(self.hps.dec_bidirectional) + 1), batch_size,\
                                                    self.hps.dec_hidden_size]))
      assert_shape_equal(encoder_outputs.size(),torch.Size([batch_size, T,\
                                                            self.hps.enc_hidden_size * (int(self.hps.dec_bidirectional) + 1)]))
    embedded = self.embedding(input)
    context = self.attn(hidden, encoder_outputs, mask)
    if __debug__:
      assert_shape_equal(context.shape, torch.Size([batch_size, self.hps.enc_hidden_size *  (int(self.hps.enc_bidirectional) + 1)]))
    assert context.shape[0] == batch_size, len(context.shape) == 2
    if translation_memory is not None: # calculate scores q
      hidden_state_from_memory, output_exp_from_memory = translation_memory.match(context)
      if __debug__ and first_iter:
        assert((hidden_state_from_memory == hidden).all(), '{}\t{}'.format(hidden.size(), hidden_state_from_memory.size()))
      # print(context.size(), hidden_state_from_memory.size())

      retrieval_gate_input = torch.cat((hidden.permute(1, 0, 2).contiguous().view(batch_size, -1),\
                                        hidden_state_from_memory.permute(1, 0, 2).contiguous().view(batch_size, -1),\
                                        context.contiguous().view(batch_size, -1)), 1)
      retrieval_gate = torch.sigmoid(translation_memory.retrieval_gate(retrieval_gate_input))

    
      is_testing = "testing" if not self.train else "training"
      #print(is_testing)

      self.writer.add_scalar("translation_memory/retrieval_gate_mean_{}".format(is_testing), retrieval_gate.mean().data.cpu().numpy()[0], self.i)
      self.writer.add_scalar("translation_memory/retrieval_gate_max_{}".format(is_testing), retrieval_gate.max().data.cpu().numpy()[0], self.i)
      self.writer.add_scalar("translation_memory/retrieval_gate_min_{}".format(is_testing), retrieval_gate.min().data.cpu().numpy()[0], self.i)

    if translation_memory is not None and self.hps.dec_use_deep_fusion:
      hidden = retrieval_gate * hidden + (1 - retrieval_gate) * hidden_state_from_memory
    rnn_input = torch.cat((embedded, context), -1).view(batch_size, 1, -1)

    output, next_hidden = self.gru(rnn_input, hidden)
    output = self.out(output).view(batch_size, self.output_size)
    output = F.log_softmax(output, -1)
    if translation_memory is not None and self.hps.dec_use_shallow_fusion:
      output = (retrieval_gate * output.clamp(-10, 10).exp() + (1 - retrieval_gate) * output_exp_from_memory.clamp(0.001, 0.999)).log()
    self.i += 1
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
      dec_bidirectional=True,
      dec_use_deep_fusion=True,
      dec_use_shallow_fusion=False
    )
