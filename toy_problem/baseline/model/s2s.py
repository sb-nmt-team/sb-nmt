import torch
import numpy as np

from torch import nn
from torch.autograd import Variable

from data import lang
from utils.hparams import merge_hparams
from model.encoder import EncoderRNN
from model.decoder import DecoderRNN

from searchengine.searchengine import SearchEngine

class Search(object):
    def __init__(self, s2s, top_size=3):
        self.searchengine = SearchEngine()
        self.searchengine.load("se.bin")
        self.s2s = s2s
        self.src_lang = s2s.source_lang
        self.trg_lang = s2s.target_lang
        self.top_size = top_size

    def init_search_engine(self, input_sentences):
        search_inputs, search_outputs = [], []
        for sentence in input_sentences:
            found_inputs = self.searchengine(sentence, n_neighbours=self.top_size)
            found_outputs = list(map(self.database.get, found_inputs))
            assert(len(search_results) == self.top_size)
            search_inputs += found_inputs
            search_outputs += found_outputs
            
        search_inputs, input_mask = self.src_lang.convert_batch(search_inputs)
		search_outputs, output_mask = self.src_lang.convert_batch(search_outputs)
        
        batch_size, sentence_length, hidden_size = search_inputs.shape
        search_inputs = batch.view(batch_size, 1, sentence_length, hidden_size)\
            .expand(batch_size, self.top_size, sentence_length, hidden_size).contiguous()\
            .view(search_batches.shape)
        self.hiddens, self.contexts = s2s.get_hiddens_and_contexts(search_inputs, input_bask, search_batches, output_mask)
        
    
    def match(self, context):
        '''
        context = Variable(FloatTensor(B, H))
        '''
        B, H = context.shape
        energies = torh.billinear(context.view(B, 1, H).expand(B, self.top_size, H), context, 1)
        energies = torch.softmax(energies, dim=1)
        hidden = energies.dot(self.hiddens)
        output = energies.dot(self.outputs)
        return hidden, output
        
class Seq2Seq(nn.Module):
	def __init__(self, source_lang, target_lang, hps, training_hps):
		super(Seq2Seq, self).__init__()
		self.hps = hps
		self.training_hps = training_hps
		self.source_lang = source_lang
		self.target_lang = target_lang

		self.encoder = EncoderRNN(source_lang.input_size(), self.hps, self.training_hps)
		self.decoder = DecoderRNN(target_lang.input_size(), target_lang.input_size(), self.hps, self.training_hps)

		self.max_length = self.training_hps.max_length
		self.criterion = nn.NLLLoss(reduce=False, size_average=False)
        self.search = Search(self)

	def translate(self, input_batch, mask, use_search=False):
		batch_size = input_batch.size()[0]
		encoder_outputs = self.encoder(input_batch)
        if use_search:
            search_engine = self.search.get_search_engine(input_batch)
        
        
        
		hidden = None

		dec_input = Variable(torch.LongTensor([lang.BOS_TOKEN] * batch_size))

		if self.training_hps.use_cuda:
			dec_input = dec_input.cuda()

		translations = [[lang.BOS_TOKEN] for _ in range(batch_size)]
		converged = np.zeros(shape=(batch_size,))
		for i in range(self.max_length):
            if use_search:
                output, hidden = self.decoder(dec_input, decoder_outputs, mask=mask, hidden=hidden, search_engine=search_engine)
            else:
                output, hidden = self.decoder(dec_input, encoder_outputs, mask=mask, hidden=hidden)
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
			output, hidden = self.decoder(output_batch[:, i], encoder_outputs, mask=mask, hidden=hidden)
			loss += (self.criterion(output, output_batch[:, i + 1]) * out_mask[:, i + 1]).sum()

		loss /= out_mask.sum()
		return loss
    
    def get_hiddens(self, input_batch, mask, output_batch, out_mask):
		encoder_outputs = self.encoder(input_batch)

		hidden = None

		loss = 0.0
        hiddens = []
		for i in range(out_mask.size()[1] - 1):
			output, hidden = self.decoder(output_batch[:, i], encoder_outputs, mask=mask, hidden=hidden)
        assert(len(hidden.shape) == 3)
        hiddens.append(hidden)
        assert(len(hiddens.shape) == 4)
		return torch.cat(hiddens)


	@staticmethod
	def get_default_hparams():
		return merge_hparams(EncoderRNN.get_default_hparams(), DecoderRNN.get_default_hparams())