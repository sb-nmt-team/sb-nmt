import pickle
import sys
from .utils import find_closest
from utils.launch_utils import log_func

class SearchEngine(object):
  @log_func
  def __init__(self):
    self.dataset = []
    self.nn_map = {}
    self.n_neighbours = None
    self.dictionary = {}

  @log_func
  def load(self, path):
    with open(path, "rb") as file:
      content = pickle.load(file)
      self.dataset = content["dataset"]
      self.nn_map = content["result"]
      if self.n_neighbours is None:
        self.n_neighbours = len(self.dataset)

      if self.nn_map:
        some_key = next(self.nn_map.__iter__())
        self.n_neighbours = min(self.n_neighbours,
                                len(self.nn_map[some_key]))

  def not_found_warning(self, sentence):
    pass

  def sentence_to_tuple(self, sentence):
    if isinstance(sentence, str):
      sentence = tuple(sentence.split())
    elif isinstance(sentence, list):
      sentence = tuple(sentence)
    return sentence

  def indexes_to_sentences(self, indexes):
    return [(score, self.dataset[index]) for score, index in indexes]

  def set_dictionary(self, dataset):
    """
    :param dataset: (result of data.lang.read_problem)[0]
    """

    for mode, mode_dataset in dataset.items():
      src, tgt = mode_dataset
      for (src_sent, tgt_sent) in zip(src, tgt):
        self.dictionary[self.sentence_to_tuple(src_sent)] = \
          self.sentence_to_tuple(tgt_sent)

  @log_func
  def add_translation(self, result):
    if not all([sentence in self.dictionary for _, sentence in result]):
      print("Not all sentences have translation in dictionary, " +
            "fallback to translation=False",
            file=sys.stderr)
      return result
    return [(score, sentence, self.dictionary[sentence])
            for score, sentence in result]

  @log_func
  def __call__(self, sentence, n_neighbours=None, translation=False):
    """
    for translation=True to work, set_dictionary must have been called
    :return list of (fuzzy_score, source_sent, target_sent) if translation=True
      or list of (fuzzy_score, source_sent) if translation=False,
      where fuzzy_score is float in [0.0, 1.0], and *_sent is tuple of tokens
    """
    sentence = self.sentence_to_tuple(sentence)

    if sentence in self.nn_map:
      result = self.nn_map[sentence]
      if n_neighbours:
        result = result[:n_neighbours]
      return self.indexes_to_sentences(result)

    self.not_found_warning(sentence)

    if not n_neighbours:
      n_neighbours = self.n_neighbours

    result = self.indexes_to_sentences(
      find_closest(sentence, self.dataset, n_neighbours)
    )

    if translation:
      result = self.add_translation(result)
    return result



