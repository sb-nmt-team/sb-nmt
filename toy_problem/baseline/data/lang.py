import os
import numpy as np
from collections import defaultdict

PAD_TOKEN = 0
BOS_TOKEN = 1
NAN_TOKEN = 2
EOS_TOKEN = 3
SPECIAL_TOKENS = 4
OCCURING_SPECIAL_TOKENS = 1


def read_file(filename):
    with open(filename) as file:
        return list(map(lambda s: s.strip().split(" "), file.readlines()))

def read_problem(path, n_sents=None):
  modes = ["train", "dev", "test"]
  file_template = "{}.{}.txt"

  result = {}
  for mode in modes:
    src = read_file(os.path.join(path, file_template.format("src", mode)))
    tgt = read_file(os.path.join(path, file_template.format("tgt", mode)))

    assert len(src) == len(tgt)
    if n_sents is not None:
      result[mode] = (src[:n_sents], tgt[:n_sents])
    else:
      result[mode] = (src, tgt)

  src_lang = Lang(os.path.join(path, file_template.format("src", "tokens")))
  tgt_lang = Lang(os.path.join(path, file_template.format("tgt", "tokens")))
  return result, src_lang, tgt_lang


class Lang:
  def __init__(self, tokens_file_path):
    self.idx2word = defaultdict(lambda: "<NAN/>")
    self.word2idx = defaultdict(lambda: NAN_TOKEN)
    with open(tokens_file_path) as tokens_file:
      tokens = tokens_file.readlines()
      for word, idx in map(lambda x: x.strip().split(), tokens):
        idx = int(idx) + SPECIAL_TOKENS
        self.idx2word[idx] = word
        self.word2idx[word] = idx
      assert PAD_TOKEN not in self.idx2word
      assert BOS_TOKEN not in self.idx2word
      assert EOS_TOKEN not in self.idx2word
      for word, idx in [('<PAD/>', PAD_TOKEN), ('<S>', BOS_TOKEN),
                        ('</S>', EOS_TOKEN), ('<NAN/>', NAN_TOKEN)]:
        self.idx2word[idx] = word
        self.word2idx[word] = idx

  def convert(self, sentence):
    if isinstance(sentence, str):
      sentence = sentence.strip().split()
    return [BOS_TOKEN] + list(map(lambda word: self.word2idx[word], sentence)) + [EOS_TOKEN]

  def convert_batch(self, sents):

    batch_max_length = 0
    for sent in sents:
      batch_max_length = max(batch_max_length, len(sent))

    result = np.zeros(shape=(len(sents), batch_max_length + 1 + 1))
    mask = np.zeros(shape=(len(sents), batch_max_length + 1 + 1))

    for sent_id, sent in enumerate(sents):
      sent = sent[:batch_max_length]
      current = self.convert(sent)
      result[sent_id, :len(current)] = current
      mask[sent_id, :len(current)] = 1.0

    return result, mask

  def input_size(self):
    return len(self.idx2word.keys())

  def output_size(self):
    return len(self.idx2word.keys())

  def get_word(self, idx):
    return self.idx2word[idx]

  def get_eos(self):
    return EOS_TOKEN