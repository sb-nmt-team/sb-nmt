from search_engine.searchengine import SearchEngine

class TranslationMemory(object):

  def __init__(self, s2s, database, top_size=3):
    self.searchengine = SearchEngine()
    self.searchengine.load("../search_engine/se.bin")
    self.s2s = s2s
    self.source_lang = s2s.source_lang
    self.target_lang = s2s.target_lang
    self.top_size = top_size
    self.database = dict(database)

  def fit(self, input_sentences):
    search_inputs, search_outputs = [], []
    for sentence in input_sentences:
      sentenсe = ' '.join(map(self.target_lang.get_word, sentence))
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
    energies = (context.view(B, 1, H).expand(B, self.top_size, H) *  context).sum(dim=2)
    energies = torch.softmax(energies, dim=1)
    hidden = energies.dot(self.hiddens)
    output = energies.dot(self.outputs)
    return hidden, output