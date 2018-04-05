import numpy as np
import unicodedata
import collections

import matplotlib.pyplot as plt
from collections import Counter, defaultdict
import string
import sys
import pickle

bad_characters = set(string.punctuation + string.digits)

def read_dataset(dataset_name):
    dataset = defaultdict(list)
    rtlm = "\u200F"
    bos = '_'
    eos = ';'
    
    
    en_characters = set(["<\s>"])
    he_characters = set(["<\s>"])
    
    with open(dataset_name, "r") as fin:
        for line in fin:
            en,he = line.strip().lower().replace(bos,' ').replace(eos,' ').replace(rtlm, '').split('\t')
            word, trans = he, en
            en_characters |=  set(en)
            he_characters |= set(he)

#             if len(word) < 3: continue
#             if EASY_MODE:
#                 if max(len(word),len(trans))>20:
#                     continue

            dataset[word].append(trans)
    
    en_characters = list(sorted(list(en_characters)))
    he_characters = list(sorted(list(he_characters)))
    # print ("size = ",len(dataset))
    return dataset, en_characters, he_characters


def filter_multiple_translations(data):
    result = {}
    for original, translations in data.items():
        if len(set(translations)) > 1:
            continue
        if ' ' in translations[0]:
            continue
        result[original] = translations[0]
    # print ("size = ",len(result))
    return result


def filter_bad_characters(data, bad_characters):
    result = {}
    for source, translation in data.items():
        assert isinstance(translation, str)
        if len(set(source) & bad_characters) != 0:
                continue
        if len(set(translation) & bad_characters) != 0:
                continue
        result[source] = translation
    
    # print ("size = ",len(result))
    return result


def filter_copied_words(data):
    result = {}
    for original, translation in data.items():
        assert isinstance(translation, str)
        if len(set(original) & set(translation)) > 0:
            continue
        result[original] = translation
    # print ("size = ",len(result))
    return result


def filter_short_targets(data, target_threshold=2):
    result = {}
    for original, translation in data.items():
        assert isinstance(translation, str)
        if len(translation) <= target_threshold:
            continue
        result[original] = translation
    
    # print ("size = ",len(result))
    return result


def split_train_test(data, valid_size=0.1, test_size=0.1):
    assert valid_size >= 0 and valid_size < 1.0
    assert test_size >= 0 and test_size < 1.0
    
    train_size  = 1.0 - (valid_size + test_size)
    
    np.random.seed(42)
    
    
    sources = np.array(list(data.keys()))
#     print(sources)
    n_sources = len(sources)
    index_permutation = np.random.permutation(np.arange(n_sources))
    
    train_border = int(n_sources * train_size)
    valid_border = int(n_sources * (train_size + valid_size))
    train_sources = sources[index_permutation[: train_border]]
    valid_sources = sources[index_permutation[train_border : valid_border]]
    test_sources = sources[index_permutation[valid_border :]]
    
    def create_dataset(sources):
        result = {}
        for source in sources:
            result[source] = data[source]
        return result
    
    
    return create_dataset(train_sources), create_dataset(valid_sources), create_dataset(test_sources)


def pipeline(dataset):
    filtered_dataset = filter_multiple_translations(dataset)
    f1 = filter_short_targets(filtered_dataset)
    f2 = filter_bad_characters(f1, bad_characters)
    f3 = filter_copied_words(f2)
    f4 = filter_bad_characters(f3, bad_characters | set([' ']))
    
    
    return [dataset, filtered_dataset, f1, f2, f3, f4]


def transformation_pipeline(dataset, valid_size=0.1, test_size=0.1):
    train, valid, test = split_train_test(dataset)
    trains = pipeline(train)
    valids = pipeline(valid)
    tests = pipeline(test)
    
    # plt.title('Dataset changes with refinement')
    # plt.plot([len(x) for x in trains], label='train')
    # plt.plot([len(x) for x in valids], label='valid')
    # plt.plot([len(x) for x in tests], label='test')
    # plt.legend()
    # plt.show()
    
    return trains, valids, tests
    
    
if __name__ == "__main__":
    dataset, en_characters, he_characters = read_dataset(sys.argv[1])
    trains, valids, tests = transformation_pipeline(dataset)

    result = {
        "data" : dict(trains=trains, valids=valids, tests=tests),
        "characters": dict(en=en_characters, he=he_characters)
    }

    pickle.dump(result, open(sys.argv[2], "wb"))
