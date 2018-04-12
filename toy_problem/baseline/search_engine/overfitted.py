from search_engine.searchengine import SearchEngine


class OverfittedSearchEngine(SearchEngine):
  def __init__(self):
    super().__init__()

  def __call__(self, sentence, n_neighbours=None, translation=False):
    sentence = self.sentence_to_tuple(sentence)
    result = super().__call__(sentence, n_neighbours=n_neighbours, translation=False)
    result = [(1.0, sentence)] + result[:-1]
    if translation:
      result = self.add_translation(result)
    return result
