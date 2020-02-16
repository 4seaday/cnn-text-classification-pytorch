"""
tokenize

build_vocab

"""
import os
from pathlib import Path
from tqdm import tqdm
import pandas as pd
from soynlp.word import WordExtractor
from soynlp.tokenizer import MaxScoreTokenizer
import json
import pickle

import torch
import itertools
from collections import Counter
from typing import List, Dict

class Params:

    def __init__(self, json_path):
        self.update(json_path)
        self.cuda_is_avail()

    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def cuda_is_avail(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



        params = {'device': device}
        self.__dict__.update(params)

    @property
    def dict(self):
        return self.__dict__



def build_tokenizer():

    data_dir = Path().cwd() / 'data'
    corpus_file = os.path.join(data_dir, 'ratings.txt')

    df = pd.read_table(corpus_file, encoding='utf-8')
    # df = df.head(2000)

    word_extractor = WordExtractor()
    word_extractor.train(str(df['document']))
    words = word_extractor.extract()
    cohesion_score = {word:score.cohesion_forward for word, score in words.items()}
    tokenizer = MaxScoreTokenizer(scores=cohesion_score)

    with open('pickles/tokenizer.pickle', 'wb') as pickle_out:
        pickle.dump(tokenizer, pickle_out)


def build_vocab():

    pickle_tokenizer = open('pickles/tokenizer.pickle', 'rb')
    tokenizer = pickle.load(pickle_tokenizer)

    # if config.mode == 'train':
    #     mode = 'train'
    # else:
    #     mode = 'test'

    data_dir = Path().cwd() / 'data'
    sentence_file = os.path.join(data_dir, 'ratings.txt')
    df = pd.read_table(sentence_file, encoding='utf-8')

    sentence_list = []
    for _, row in tqdm(df.iterrows()):
        sentence_list.append(tokenizer.tokenize(str(row[1])))

    token_counter = Counter(itertools.chain.from_iterable(sentence_list))

    vocab = ['<PAD>'] + [word for word in token_counter.keys()]
    vocab_size = len(vocab)
    print('Total Vocab size : ', vocab_size)

    word_to_idx = {idx: word for word, idx in enumerate(vocab)}

    with open('pickles/vocab.pickle', 'wb') as vocabulary:
        pickle.dump(word_to_idx, vocabulary)


def padding_sentence(config):
    pickle_tokenizer = open('pickles/tokenizer.pickle', 'rb')
    tokenizer = pickle.load(pickle_tokenizer)

    data_dir = Path().cwd() / 'data'

    if config.mode == 'train':
        mode = 'train'
    else:
        mode = 'test'

    corpus_file = os.path.join(data_dir, f'ratings_{mode}.txt')
    df = pd.read_table(corpus_file, encoding='utf-8')

    sentence_list = []
    for _, row in tqdm(df.iterrows()):
        sentence_list.append(tokenizer.tokenize(str(row[1])))


    max_sequence_length = max(len(sentence) for sentence in sentence_list)
    ################################ train, test 모두 동일해야되는거 아닌가??

    padded_sentence = []
    for sentence in sentence_list:
        temp_len = len(sentence)
        padded_sentence.append(sentence + ['<PAD>' for _ in range(max_sequence_length - temp_len)])

    return padded_sentence


def token_to_idx(sentence_list: List[List[str]], vocab: Dict[str, int]) -> List[List[int]]:

    input_sentence = []
    for sentence in sentence_list:
        temp_list = []
        print(sentence)
        for token in sentence:
            temp_list.append(vocab[token])
        input_sentence.append(temp_list)

    return input_sentence
