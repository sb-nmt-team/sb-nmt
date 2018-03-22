import pickle
from utils import find_closest

class SearchEngine(object):
    def __init__(self):
        self.dataset = []
        self.nn_map = {}
        self.n_neighbours = None

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

    def indexes_to_sentences(self, indexes):
        return [(score, self.dataset[index]) for score, index in indexes]

    def __call__(self, sentence, n_neighbours=None):
        if isinstance(sentence, str):
            sentence = tuple(sentence.split())
        elif isinstance(sentence, list):
            sentence = tuple(sentence)
        
        print(sentence)

        if sentence in self.nn_map:
            result = self.nn_map[sentence]
            if n_neighbours:
                result = result[:n_neighbours]
            return self.indexes_to_sentences(result)

        self.not_found_warning(sentence)

        if not n_neighbours:
            n_neighbours = self.n_neighbours

        return self.indexes_to_sentences(
            find_closest(sentence, self.dataset, n_neighbours)
        )



