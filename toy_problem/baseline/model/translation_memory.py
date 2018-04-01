from search_engine.searchengine import SearchEngine
from torch.autograd import Variable
import torch
import numpy as np

class TranslationMemory(object):

  def __init__(self, s2s, database, top_size=3, search_engine=None):
    if search_engine == None:
      search_engine = SearchEngine()
      search_engine.load("../search_engine/se.bin")
    self.searchengine = search_engine
    self.s2s = s2s
    self.source_lang = s2s.source_lang
    self.target_lang = s2s.target_lang
    self.top_size = top_size
    self.database = dict(database)

  def fit(self, input_sentences):
    batch_size = len(input_sentences)
    search_inputs, search_outputs = [], []
    for sentence in input_sentences.data.numpy():
      sentence = self.source_lang.convert(sentence, backward=True)
      found_inputs = [x[1] for x in self.searchengine(sentence, n_neighbours=self.top_size)]
      found_outputs = list(map(self.database.get, found_inputs))
#       print('(DEBUG) self.searchengine(sentence, n_neighbours=self.top_size):', self.searchengine(sentence))
#       print('(DEBUG) sentence', sentence)
#       print('(DEBUG) len(found_outputs)', len(found_outputs))
#       print('(DEBUG) found_outputs', found_outputs)
      assert(len(found_outputs) == self.top_size)
      search_inputs += found_inputs
      search_outputs += found_outputs

    search_inputs, input_mask = self.source_lang.convert_batch(search_inputs)
    search_inputs = Variable(torch.from_numpy(search_inputs.astype(np.int64))).contiguous()
    input_mask = Variable(torch.from_numpy(input_mask.astype(np.float32))).contiguous()
    search_outputs, output_mask = self.target_lang.convert_batch(search_outputs)
    search_outputs = Variable(torch.from_numpy(search_outputs.astype(np.int64))).contiguous()
    output_mask = Variable(torch.from_numpy(output_mask.astype(np.float32))).contiguous()
# #     print('(DEBUG) search_inputs.shape:',search_inputs.shape)
#     print('(DEBUG) search_outputs.shape:',search_outputs.shape)
    self.hiddens, self.contexts = self.s2s.get_hiddens_and_contexts(search_inputs, input_mask, search_outputs, output_mask)
    self.hiddens = self.hiddens.view(batch_size, -1,\
                                     self.s2s.hps.dec_layers * (self.s2s.hps.dec_bidirectional + 1),\
                                     self.s2s.hps.dec_hidden_size)
#     input_mask = input_mask.view(batch_size, -1)
    self.contexts = self.contexts.view(batch_size, -1,\
                                     self.s2s.hps.enc_layers * (self.s2s.hps.enc_bidirectional + 1),\
                                     self.s2s.hps.enc_hidden_size)
#     output_mask = output_mask.view(batch_size, -1)
    
#     search_inputs = search_inputs.view(batch_size, self.top_size, -1)
#     search_outputs = search_outputs.view(batch_size, self.top_size, -1)
#     return search_inputs, input_mask, search_outputs, output_mask


  def match(self, context):
    '''
    context = Variable(FloatTensor(B, H))
    '''
    B,  H = context.shape
    context = context.contiguous().view(B, 1, -1, self.s2s.hps.enc_hidden_size).contiguous()
    energies = (context *  self.contexts)
    energies = energies.contiguous().sum(dim=2).sum(dim=2)
    energies = torch.nn.Softmax(dim=1)(energies)
    hidden = (energies.view(B, -1, 1, 1) * self.hiddens).sum(dim=1)
#     output = (energies.view(B, -1, 1, 1) * self.outputs).sum(dim=1)
    return hidden.permute(1,0,2) #, output