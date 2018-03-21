import torch
from torch import nn
from torch.autograd import Variable

from data import lang
from utils.hparams import HParams
# fix it


class EncoderRNN(nn.Module):
	def __init__(self, input_size, hps, training_hps):
		super(EncoderRNN, self).__init__()
		self.hps = hps
		self.training_hps = training_hps
		self.num_directions = (int(hps.enc_bidirectional) + 1)

		self.embedding = nn.Embedding(num_embeddings=input_size,
		                              embedding_dim=hps.enc_emb_size,
		                              padding_idx=lang.PAD_TOKEN)

		self.gru = nn.GRU(input_size=hps.enc_emb_size,
		                  hidden_size=hps.enc_hidden_size,
		                  batch_first=True,
		                  dropout=hps.enc_dropout,
		                  num_layers=hps.enc_layers,
		                  bidirectional=hps.enc_bidirectional)

	def forward(self, input_batch, hidden=None):
		"""
		"""

		if hidden is None:
			hidden = self.init_hidden(input_batch.size(0))

		embedded = self.embedding(input_batch).contiguous()
		outputs, _ = self.gru(embedded, hidden)
		return outputs

	def init_hidden(self, batch_size):
		result = Variable(torch.zeros(self.hps.enc_layers * self.num_directions,
		                              batch_size,
		                              self.hps.enc_hidden_size))

		if self.training_hps.use_cuda:
			return result.cuda()
		else:
			return result

	@staticmethod
	def get_default_hparams():
		return HParams(
			enc_emb_size=128,
			enc_hidden_size=128,
			enc_dropout=0.1,
			enc_layers=1,
			enc_bidirectional=True
		)