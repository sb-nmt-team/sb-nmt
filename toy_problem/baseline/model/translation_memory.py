from search_engine.searchengine import SearchEngine
from torch.autograd import Variable
import torch
import os
import numpy as np
from utils.hparams import HParams
from torch.nn import Parameter
import torch.nn as nn
from utils.launch_utils import log_func
from utils.debug_utils import assert_shape_equal
from data.lang import read_problem

SE_DIR = os.path.join(os.path.abspath(os.path.join(__file__ ,"../../")), "search_engine")
DATASET_DIR = os.path.join(os.path.abspath(os.path.join(__file__ ,"../../..")), "preprocessed")


class TranslationMemory(object):
  @log_func
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
    is_bidir = int(hps.enc_bidirectional) + 1
    size = hps.enc_hidden_size * is_bidir
    #M_inits = torch.randn(size, size) * 0.01 +\
    #                   torch.eye(size).view(1, 1, size, size)
    M_inits = torch.randn(size, size) * 0.01 +\
                       torch.eye(size)
    self.M = Variable(M_inits, requires_grad=True)
    self.retrieval_gate = nn.Linear(self.hps.enc_hidden_size * (int(self.hps.enc_bidirectional) + 1) + \
                                 self.hps.dec_layers * self.hps.dec_hidden_size * (int(self.hps.dec_bidirectional) + 1) +\
                                 self.hps.dec_layers * self.hps.dec_hidden_size * (int(self.hps.dec_bidirectional) + 1), 1)

  @log_func
  def fit(self, input_sentences):
    batch_size = len(input_sentences)
    search_inputs, search_outputs = [], []
    input_sentences = input_sentences.clone()
    for sentence in input_sentences.data.cpu().numpy():
      sentence = self.source_lang.convert(sentence, backward=True)
      found = self.searchengine(sentence, n_neighbours=self.top_size, translation=True)
      found_inputs = [x[1] for x in found]
      found_outputs = [x[2] for x in found]
      # getter = lambda s: self.database.get(s, "n o t f o u n d")
      # found_outputs = list(map(getter, found_inputs))
      assert(len(found_outputs) == self.top_size)
      search_inputs += found_inputs
      search_outputs += found_outputs

    search_inputs, input_mask = self.source_lang.convert_batch(search_inputs)
    search_inputs = Variable(torch.from_numpy(search_inputs.astype(np.int64))).contiguous()
    input_mask = Variable(torch.from_numpy(input_mask.astype(np.float32))).contiguous()
    search_outputs, output_mask = self.target_lang.convert_batch(search_outputs)
    search_outputs = Variable(torch.from_numpy(search_outputs.astype(np.int64))).contiguous()
    # print(search_inputs, search_outputs)
    output_mask = Variable(torch.from_numpy(output_mask.astype(np.float32))).contiguous()
    if self.is_cuda:
        search_inputs = search_inputs.cuda()
        input_mask = input_mask.cuda()
        search_outputs = search_outputs.cuda()
        output_mask = output_mask.cuda() #
    # print(search_outputs.size())
    self.hiddens, self.contexts = self.model.get_hiddens_and_contexts(search_inputs, input_mask, search_outputs, output_mask)
    batch_size_, top_size_, max_output_length = search_outputs.view(batch_size, self.top_size, -1).shape
    assert batch_size == batch_size_
    assert top_size_ == self.top_size
    search_outputs_ohe = Variable(torch.FloatTensor(batch_size, self.top_size * (max_output_length - 1),self.target_lang.output_size()))
    search_outputs_ohe.zero_()
    search_outputs_ohe.scatter_(2, search_outputs[:, :-1].contiguous().view(batch_size, self.top_size * (max_output_length - 1), 1), 1)
    self.outputs_exp = search_outputs_ohe

    self.contexts = self.contexts.detach()
    self.hiddens = self.hiddens.detach()
    self.output_mask = output_mask[:, :-1]
    if self.is_cuda:
        self.contexts = self.contexts.cuda()
        self.hiddens = self.hiddens.cuda()

  def parameters(self):
    #return []
    yield self.M
    for param in self.retrieval_gate.parameters():
      yield param

  @log_func
  def state_dict(self, destination=None, prefix='', keep_vars=False):
    if not keep_vars:
        M = self.M.clone().data
    else:
        M = self.M
    


    name = prefix + "translation_memory." + "M"
    
    if destination is None:
        destination = {}
    
    destination[name] = M

    return destination

  @log_func
  def load_state_dict(self, state_dict, strict=True):
    name = list(self.state_dict().keys())[0]

    M = state_dict[name]
    self.M = Variable(M, requires_grad=False) 


  @log_func
  def match(self, context):
    '''
    context = Variable(FloatTensor(B, HE * DE))
    '''
    B = context.size(0)
    T = self.contexts.size(1)
    context = context.matmul(self.M) # [B, HE * DE]
    energies = self.contexts.view(B, self.hps.tm_top_size, T, self.hps.enc_hidden_size * (int(self.hps.enc_bidirectional) + 1))\
            .matmul(context.view(-1, 1, self.hps.enc_hidden_size * (int(self.hps.enc_bidirectional) + 1), 1)) # [B, self.T]
    energies = energies.view(B, self.hps.tm_top_size * T)

    energies = torch.nn.Softmax(dim=1)(energies)

    hidden = (energies.permute(1,0).contiguous().view(1, self.hps.tm_top_size * T, B, 1)\
              * self.hiddens.view(\
              self.hps.dec_layers * (int(self.hps.dec_bidirectional) + 1), self.hps.tm_top_size * T, B, self.hps.dec_hidden_size))\
      .sum(dim=1) # [LD * BD, B, HD]

    output_exp = (energies.view(B, self.hps.tm_top_size * T, 1) * self.outputs_exp).sum(dim=1) # [B, target_lang_size]
    if __debug__:
      assert_shape_equal(hidden.size(), torch.Size([self.hps.dec_layers * (int(self.hps.dec_bidirectional) + 1), B, self.hps.dec_hidden_size]))
      assert_shape_equal(output_exp.size(), torch.Size([B, self.outputs_exp.size(-1)]))
    return hidden , output_exp

  @log_func
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
      tm_overfitted = False,
      tm_bin_path = os.path.join(SE_DIR, "se.bin"),
      tm_top_size = 3,
      tm_train_dataset_path = os.path.join(DATASET_DIR, "he-en")
    )
