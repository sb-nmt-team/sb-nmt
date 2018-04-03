from search_engine.searchengine import SearchEngine
from torch.autograd import Variable
import torch
import numpy as np
from utils.hparams import HParams
from torch.nn import Parameter
import torch.nn as nn

from data.lang import read_problem

class TranslationMemory(object):

  def __init__(self, model, hps, searchengine=None):
    self.is_cuda = False
    self.model = model
    self.hps = hps
    if searchengine == None:
      searchengine = SearchEngine()
      searchengine.load(hps.tm_bin_path)
    self.searchengine = searchengine
    self.source_lang = model.source_lang
    self.target_lang = model.target_lang
    self.top_size = hps.tm_top_size
    self.database = {tuple(x[0]): tuple(x[1]) for x in zip(*read_problem(hps.tm_train_dataset_path)[0]['train'])}
    M_inits = (torch.randn(hps.enc_hidden_size * (int(hps.enc_bidirectional) + 1),\
                                   hps.enc_hidden_size * (int(hps.enc_bidirectional) + 1)) * 0.01 +\
                       torch.eye(hps.enc_hidden_size * (int(hps.enc_bidirectional) + 1))).view(1, 1,\
                                   hps.enc_hidden_size * (int(hps.enc_bidirectional) + 1),\
                                   hps.enc_hidden_size * (int(hps.enc_bidirectional) + 1))

    #self.M = Variable(M_inits.cuda(), requires_grad=True)
    self.M = Variable(M_inits, requires_grad=True)

  def fit(self, input_sentences):
    batch_size = len(input_sentences)
    search_inputs, search_outputs = [], []
    input_sentences = input_sentences.clone()
    for sentence in input_sentences.data.cpu().numpy():
      sentence = self.source_lang.convert(sentence, backward=True)
      found_inputs = [x[1] for x in self.searchengine(sentence, n_neighbours=self.top_size)]
      found_outputs = list(map(self.database.get, found_inputs))
      assert(len(found_outputs) == self.top_size)
      search_inputs += found_inputs
      search_outputs += found_outputs

    search_inputs, input_mask = self.source_lang.convert_batch(search_inputs)
    search_inputs = Variable(torch.from_numpy(search_inputs.astype(np.int64))).contiguous()
    input_mask = Variable(torch.from_numpy(input_mask.astype(np.float32))).contiguous()
    search_outputs, output_mask = self.target_lang.convert_batch(search_outputs)
    search_outputs = Variable(torch.from_numpy(search_outputs.astype(np.int64))).contiguous()
    output_mask = Variable(torch.from_numpy(output_mask.astype(np.float32))).contiguous()
    if self.is_cuda:
        search_inputs = search_inputs.cuda()
        input_mask = input_mask.cuda()
        search_outputs = search_outputs.cuda()
        output_mask = output_mask.cuda()

    self.hiddens, self.contexts = self.model.get_hiddens_and_contexts(search_inputs, input_mask, search_outputs, output_mask)
    self.hiddens = self.hiddens.view(batch_size, -1,\
                                     self.hps.dec_layers * (self.hps.dec_bidirectional + 1),\
                                     self.hps.dec_hidden_size)
#     input_mask = input_mask.view(batch_size, -1)
    self.contexts = self.contexts.view(batch_size, -1,\
                                     self.hps.enc_layers * (self.hps.enc_bidirectional + 1) *\
                                     self.hps.enc_hidden_size)
#     output_mask = output_mask.view(batch_size, -1)
    
#     search_inputs = search_inputs.view(batch_size, self.top_size, -1)
#     search_outputs = search_outputs.view(batch_size, self.top_size, -1)
#     return search_inputs, input_mask, search_outputs, output_mask
    self.contexts = self.contexts.detach()
    self.hiddens = self.hiddens.detach()
    if self.is_cuda:
        self.contexts = self.contexts.cuda()
        self.hiddens = self.hiddens.cuda()

  def parameters(self):
    #return []
    yield self.M

  def match(self, context):
    '''
    context = Variable(FloatTensor(B, H))
    '''
    B, tm_size, H = self.contexts.shape
    context = context.contiguous().view(B, 1, H, 1).contiguous()
    energies = (context *  self.contexts.view(B, tm_size, 1, H))
    #print("energies", energies)
    #print("M", self.M)
    energies = (self.M * energies)
    energies = energies.contiguous().sum(dim=2).sum(dim=2)
    energies = torch.nn.Softmax(dim=1)(energies)
    hidden = (energies.view(B, -1, 1, 1) * self.hiddens).sum(dim=1)
#     output = (energies.view(B, -1, 1, 1) * self.outputs).sum(dim=1)
    return hidden.permute(1,0,2) #, output

  def cuda(self):
    self.is_cuda = True

    #self.M = self.M.cuda()
    self.M = Variable(self.M.data.cuda(), requires_grad=True)
    return self

  def cpu(self):
    self.is_cuda = False
    self.M = self.M.cpu()
    return self

  @staticmethod
  def get_default_hparams():
    return HParams(
      tm_init = False,
      tm_bin_path = "../search_engine/se.bin",
      tm_top_size = 3,
      tm_train_dataset_path = "../../preprocessed/he-en/"
    )
