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

	def forward(self, input, encoder_outputs, mask, hidden=None):
		"""
				input: [B,]
				encoder_outputs: [B, T, HE]
				hidden: [B, layers * directions, HD]
		"""
		batch_size = input.size(0)
		if hidden is None:
			hidden = self.init_hidden(batch_size)
		embedded = self.embedding(input)
		context = self.attn(hidden, encoder_outputs, mask).view(batch_size, -1)
		rnn_input = torch.cat((embedded, context), -1).view(batch_size, 1, -1)

		output, next_hidden = self.gru(rnn_input, hidden)
		output = self.out(output).view(batch_size, self.output_size)
		output = F.log_softmax(output, -1)

		return output, next_hidden

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

    
    
    
    
    
    
class Matcher(nn.Module):
    
    def __init__(self, hps):
        self.M = Parameter(torch.randn((hps.enc_hidden_size * (int(hps.enc_bidirectional) + 1),\
                                       hps.enc_hidden_size * (int(hps.enc_bidirectional) + 1),\
                                       1)))
    def forward(context, memories):
        '''
        Input:
        context - FloatTensor(batch_size, context_size)
        memories - contexts, hiddens, target_symbols, where
          contexts - FloatTensor(batch_size, map_size, context_size)
          hiddens - FloatTensor(batch_size, map_size, hidden_size)
          target_symbols - FloatTensor(batch_size, map_size, n_chars)
        
        '''
        contexts, hiddens, target_symbols = memories
        B, map_size, context_size = hiddens.shape
        distances = torch.bmm(contexts, context.view(B, context_size, 1))
        energies = torch.softmax(distances, dim=1).view(B, map_size)
        hidden = (energies.view(B, map_size, 1) * hiddens).sum(dim=1)
        target_symbol = (energies.view(B, map_size, 1) * target_symbols).sum(dim=1)
        return hidden, target_symbol
        
    
    
    
    
    
class DecoderRNNRetrivial(nn.Module):
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
        self.retrivial_gate = nn.Linear(self.hps.dec_emb_size + \
		                             self.hps.enc_hidden_size * (int(self.hps.enc_bidirectional) + 1) +\
		                             self.hps.enc_hidden_size * (int(self.hps.enc_bidirectional) + 1)

	def forward(self, input, encoder_outputs, mask, hidden=None):
		"""
				input: [B,]
				encoder_outputs: [B, T, HE]
				hidden: [B, layers * directions, HD]
		"""
		batch_size = input.size(0)
		if hidden is None:
			hidden = self.init_hidden(batch_size)
        
		embedded = self.embedding(input)
		context = self.attn(hidden, encoder_outputs, mask).view(batch_size, -1)
        if self.match: # calculate scores q
            hidden_state_from_memory, output_from_memory = self.match(context)
            rnn_input = torch.cat((hidden_state, hidden_state_from_memory, context), -1).view(batch_size, 1, -1)
            retrivial_gate = self.retrivial_gate(rnn_input)
        if self.deep_fusion:
            hidden_state = retrivial_gate * hidden_state + (1 - retrivial_gate) * hidden_state_from_memory
            
		rnn_input = torch.cat((embedded, context), -1).view(batch_size, 1, -1)

		output, next_hidden = self.gru(rnn_input, hidden)
		output = self.out(output).view(batch_size, self.output_size)
		output = F.log_softmax(output, -1)
        if self.shallow_fusion:
            output = retrivial_gate * output + (1 - retrivial_gate) * output_from_memory

		return output, next_hidden, 

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










































