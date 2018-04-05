import torch
import torch.nn.functional as F
from torch import nn

'''
def get_enc_output_size(self):
        return 

    def get_dec_output_size(self):
        return self.dec_hidden_size * (int(self.dec_bidirectional) + 1)

    def get_dec_state_size(self):
        return
'''

class Attn(nn.Module):
  def __init__(self, hps):
    super(Attn, self).__init__()
    self.attn = nn.Linear(hps.enc_hidden_size * (int(hps.enc_bidirectional) + 1) + \
                          hps.dec_hidden_size * (int(hps.dec_bidirectional) + 1) * hps.dec_layers,
                          1)

  def forward(self, hidden, encoder_outputs, mask):
    '''
    :param hidden:
        previous hidden state of the decoder, in shape (layers * directions, B, HD)
    :param encoder_outputs:
        encoder outputs from Encoder, in shape (B, T, HE)
    :param encoder_output_lengths:
        lengths of encoded sentences, in shape (B,)
    :return
        attention energies in shape (B,T)
    '''
    batch_size = encoder_outputs.size(0)
    max_len = encoder_outputs.size(1)

    hidden = hidden.transpose(0, 1).contiguous()  # [B, l * d, HD]
    hidden = hidden.view(batch_size, -1)  # [B, HD * layers * directions]
    hidden = hidden.repeat(max_len, 1, 1).transpose(0, 1)  # [B, T, HD * layers * directions]

    energies = self.attn(torch.cat((hidden, encoder_outputs), -1)).view(batch_size, max_len)  # [B, T, 1]

    energies = energies * mask
    energies = F.softmax(energies, dim=-1)
    energies = energies * mask
    energies = energies / energies.sum(1).view(-1, 1)  # [B, T]

    return (energies.view(batch_size, max_len, 1) * encoder_outputs).sum(1)  # [B, HE]
