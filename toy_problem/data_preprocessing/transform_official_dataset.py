import numpy as np
import unicodedata
import collections

# import matplotlib.pyplot as plt
from collections import Counter, defaultdict
import itertools
import os
import string


def read_one_file(path):
    UNICODE_SPECIAL = {
        "\u202c",
        "\u200e",
        "\u200F"
    }
    result = []
    chars = set()
#     with open("")
    with open(path, "r") as f:
        for line in f:
            line = tuple(line.strip())
#             line = line.strip()
            line = tuple(filter(lambda c: c not in UNICODE_SPECIAL, line))
#             print(line)
            result.append(line)
            chars |= set(line)
            
#     print(chars)
#     print(set(list(line)))
    return result, chars


def read_language(path, lang, replace=True):
    file_patterns = [
        "{}.train.txt",
        "{}.test.txt",
        "{}.dev.txt"
    ]
    
    replace = replace and lang in {"he", "hewv"}
    transliteration = read_transliteration_file(os.path.join(path, "hebrew_table.txt"))
    
    dataset = []
    chars = set()
    for file_pattern in file_patterns:
        data, current_chars = read_one_file(os.path.join(path, file_pattern.format(lang)))
        if replace:
                data, current_chars, transliteration = replace_hebrew(data, transliteration)
        dataset.append(data)
        chars |= current_chars
    
    if replace:
        print("Ignored:")
        for key, value in transliteration.items():
            if value == "?":
                print(key, "\t", hex(ord(key)), "\t", str(key))
        
    return dataset, list(sorted(chars))


def read_transliteration_file(file_name):
    with open(file_name, "r") as transliteration_file:
        mapping = defaultdict(lambda:"?")
        for line in transliteration_file:
            line = line.strip().split("\t")
            mapping[line[0]] = line[1]
        return mapping


def replace_hebrew(data, transliteration):
        
    result = []
    chars = set()
    for word in data:
        word = tuple(transliteration[c] for c in word)
        result.append(word)
        chars |= set(word)

    return result, set(transliteration.values()), transliteration



def filter_hebrew_in_english(data, transliteration):
    hebrew_characters = set(transliteration.keys()) - set(string.punctuation)
    result = {}
    chars = set()
    for src, tgts in data.items():
        res_tgts = []
        for tgt in tgts:
            if len(set(tgt) & hebrew_characters) == 0:
                res_tgts.append(tgt)
                chars |= set(tgt)
        result[src] = tgts
    
    print("After filtering hebrew in english dataset size = ", len(result))
    return result, list(sorted(chars))



def read_dataset(path, should_replace_hebrew, filter_english, src_lang, tgt_lang):
    assert src_lang in {"he", "hewv"}
    assert tgt_lang in {"en", "hewv"}
    he, he_chars = read_language(path, src_lang, should_replace_hebrew)
#     hewv, hewv_chars = read_language("./he-en/", "hewv", replaces)
    en, en_chars = read_language(path, tgt_lang, should_replace_hebrew)
    assert len(he) == len(en)
    
    
    result = defaultdict(list)
    
    line_counter = 0
    for part_he, part_en in zip(he, en):
#         print(part_he)
        assert len(part_he) == len(part_en)
        for src, tgt in zip(part_he, part_en):
            result[src].append(tgt)
            line_counter += 1
    
    print("Read src = {}, tgt = {}".format(src_lang, tgt_lang))
    print("\t#entries = ", len(result), " total lines = ", line_counter)
    
    if tgt_lang == 'en' and filter_english:
        result, en_chars = filter_hebrew_in_english(result, read_transliteration_file(os.path.join(path, "hebrew_table.txt")))
        
    return result, he_chars, en_chars
    

def filter_multiple_translations(data):
    result = {}
    for original, translations in data.items():
        if len(set(translations)) > 1:
            continue
        if ' ' in translations[0]:
            continue
        result[original] = translations[0]
    print ("size = ",len(result))
    return result


def split_train_test(data, valid_size=0.1, test_size=0.1, seed=42):
    assert valid_size >= 0 and valid_size < 1.0
    assert test_size >= 0 and test_size < 1.0
    
    train_size  = 1.0 - (valid_size + test_size)
    
    np.random.seed(seed)
    
    
    sources = np.array(list(data.keys()))
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
    # return [dataset, filtered_dataset]
    return filtered_dataset


def transformation_pipeline(dataset, valid_size=0.1, test_size=0.1, seed=42):
    train, valid, test = split_train_test(dataset, valid_size, test_size, seed)
    trains = pipeline(train)
    valids = pipeline(valid)
    tests = pipeline(test)
    
    return trains, valids, tests
    

def write_tokens(tokens, path):
    with open(path, "w") as f:
        for id, token in enumerate(tokens):
            f.write("{}\t{}\n".format(token, id))
     
def write_sentences(sentences, path):
    with open(path, "w") as f:
        for id, sent in enumerate(sentences):
            f.write("{}\n".format(" ".join(sent)))


if __name__ == "__main__":
    problems = {
            "hewv-en":("hewv", "en"), 
            "he-en":("he", "en"), 
            "he-hewv":("he", "hewv")
        }

    path = "./he-en/"
    seed = 42
    valid_size=0.1
    test_size=0.1
    should_replace_hebrew = True
    filter_english = True
    src_lang, tgt_lang = problems["hewv-en"]
    data, src_chars, tgt_chars = read_dataset(path, should_replace_hebrew=should_replace_hebrew, filter_english=filter_english, src_lang=src_lang, tgt_lang=tgt_lang)
    dataset = transformation_pipeline(data, valid_size, test_size, seed)

    output_dir = "./preprocessed/"
    os.makedirs(output_dir, exist_ok=True)

    file_patterns = [
        "{}.train.txt",
        "{}.dev.txt",
        "{}.test.txt"
    ]
    print(dataset[0].__class__, len(dataset[0]))

    for current_part, file_pattern in zip(dataset, file_patterns):
        src_path = os.path.join(output_dir, file_pattern.format("src"))
        tgt_path = os.path.join(output_dir, file_pattern.format("tgt"))

        current_part = sorted(list(current_part.items()))
        src_part = []
        tgt_part = []
        for src_sent, tgt_sent in current_part:
            src_part.append(src_sent)
            tgt_part.append(tgt_sent)


        write_sentences(src_part, src_path)
        write_sentences(tgt_part, tgt_path)


    write_tokens(src_chars, os.path.join(output_dir, "src.tokens.txt"))
    write_tokens(tgt_chars, os.path.join(output_dir, "tgt.tokens.txt"))



