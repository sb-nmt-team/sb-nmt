from search_engine.searchengine import SearchEngine
from torch.autograd import Variable
import torch
import os
import numpy as np
from utils.hparams import HParams
from torch.nn import Parameter
import torch.nn as nn
from utils.launch_utils import log_func
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

    #self.M = Variable(M_inits.cuda(), requires_grad=True)
    self.M = Variable(M_inits, requires_grad=True)
    #print("M size", self.M.size())

  @log_func
  def fit(self, input_sentences):
    batch_size = len(input_sentences)
    search_inputs, search_outputs = [], []
    input_sentences = input_sentences.clone()
    for sentence_num, sentence in enumerate(input_sentences.data.cpu().numpy()):
      if sentence_num == 0:
        print(sentence)
      sentence = self.source_lang.convert(sentence, backward=True)
      if sentence_num == 0:
        print(self.source_lang.convert(sentence))
      #print(sentence)
      #print(self.searchengine)
      found = self.searchengine(sentence, n_neighbours=self.top_size, translation=True)
     # print(found)
     # print("============")
      found_inputs = [x[1] for x in found]
      found_outputs = [x[2] for x in found]
      #found_inputs = [x[1] for x in self.searchengine(sentence, n_neighbours=self.top_size)]
      if sentence_num == 0:
        print(sentence)
        print(found_inputs)
        print(found_outputs)
        print("===========")
        getter = lambda s: self.database.get(s, "n o t f o u n d")
        found_outputs = list(map(getter, found_inputs))
        print(found_outputs)
        print("%%%%%%%")
      assert(len(found_outputs) == self.top_size)
      search_inputs += found_inputs
      search_outputs += found_outputs
    #print(search_inputs[0])
    search_inputs, input_mask = self.source_lang.convert_batch(search_inputs)
    #print(search_inputs[0])
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
    #print("Hiddens, contexts", self.hiddens.size(), self.contexts.size())
    self.hiddens = self.hiddens.view(batch_size, -1,\
                                     self.hps.dec_layers * (self.hps.dec_bidirectional + 1),\
                                     self.hps.dec_hidden_size)
#     input_mask = input_mask.view(batch_size, -1)
    #print(self.contexts[0, 0:3, :10])
    self.contexts = self.contexts.view(batch_size, -1,\
                                     self.hps.enc_layers * (self.hps.enc_bidirectional + 1) *\
                                     self.hps.enc_hidden_size)
    #print(self.contexts[0, 0:3, :10])
#     output_mask = output_mask.view(batch_size, -1)
    
#     search_inputs = search_inputs.view(batch_size, self.top_size, -1)
#     search_outputs = search_outputs.view(batch_size, self.top_size, -1)
#     return search_inputs, input_mask, search_outputs, output_mask
    self.contexts = self.contexts.detach()
    self.hiddens = self.hiddens.detach()
    self.output_mask = output_mask[:, :-1]
    if self.is_cuda:
        self.contexts = self.contexts.cuda()
        self.hiddens = self.hiddens.cuda()

  def parameters(self):
    #return []
    yield self.M

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



    """
  def match(self, context):
    '''
    context = Variable(FloatTensor(B, H))
    '''
    B, tm_size, H = self.contexts.shape
    context = context.contiguous().view(B, 1, H, 1).contiguous()
    energies = (context *  self.contexts.view(B, tm_size, 1, H))
    #print(self.contexts.size())
    #print("energies", energies)
    #print("M", self.M)
    energies = (self.M * energies)
    energies = energies.contiguous().sum(dim=2).sum(dim=2)
    energies = torch.nn.Softmax(dim=1)(energies)
    hidden = (energies.view(B, -1, 1, 1) * self.hiddens).sum(dim=1)
#     output = (energies.view(B, -1, 1, 1) * self.outputs).sum(dim=1)
    return hidden.permute(1,0,2) #, output
  """

  @log_func
  def match(self, context, verbose=False):
    '''
    context = Variable(FloatTensor(B, H))
    '''
    ppring = print
    pprint = lambda *args, **kwargs: ppring(*args, **kwargs) if verbose else None
    B, tm_size, H = self.contexts.shape
    #print(self.M.size(), context.size())
    context = context.matmul(self.M)
    pprint(context.view(B, 1, H)[0, 0, :10])
    pprint(self.contexts.view(B, tm_size, H)[0, 0:2, :10])
    #assert False
    context = context.contiguous().view(B, 1, H).contiguous()
    energies = (context *  self.contexts.view(B, tm_size, H))
    #print(self.contexts.size())
    pprint("output_mask", self.output_mask.size())
    #print(self.output_mask)
    #print("M", self.M)
    #energies = (self.M * energies)
    energies = energies.contiguous().sum(dim=2)
    energies = torch.nn.Softmax(dim=1)(energies)
    pprint("energies", energies.size())
    energies = energies * self.output_mask
    if verbose:
      pprint(energies)
    normalizer = energies.sum(dim=1).unsqueeze(1)
    energies /= normalizer
    if verbose:
      pprint("-----")
      pprint(energies)
    pprint("Energies")
    #print(energies.max(dim=1))
    pprint(self.hiddens.size())
    hidden = (energies.view(B, -1, 1, 1) * self.hiddens).sum(dim=1)
#     output = (energies.view(B, -1, 1, 1) * self.outputs).sum(dim=1)
    return hidden.permute(1,0,2) #, output

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
