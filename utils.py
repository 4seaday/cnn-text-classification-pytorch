"""
tokenize

build_vocab

"""
import os
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import numpy as np
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
        self.build_vocab()

    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def build_vocab(self):
        pickle_tokenizer = open('pickles/tokenizer.pickle', 'rb')
        tokenizer = pickle.load(pickle_tokenizer)

        data_dir = Path().cwd() / 'data'
        # sentence_file = os.path.join(data_dir, 'ratings.txt')
        sentence_file = os.path.join(data_dir, 'ratings_train.txt')
        df = pd.read_table(sentence_file, encoding='utf-8')
        # df = df.head(2000)

        sentence_list = []
        for _, row in tqdm(df.iterrows()):
            sentence_list.append(tokenizer.tokenize(str(row[1])))

        max_sequence_length = max(len(sentence) for sentence in sentence_list)

        token_counter = Counter(itertools.chain.from_iterable(sentence_list))

        vocab = ['<PAD>'] + [word for word in token_counter.keys()]
        vocab_size = len(vocab)
        print('Total Vocab size : ', vocab_size)

        word_to_idx = {idx: word for word, idx in enumerate(vocab)}

        with open('pickles/vocab.pickle', 'wb') as vocabulary:
            pickle.dump(word_to_idx, vocabulary)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        params = {'device': device, 'max_sequence_length': max_sequence_length,
                  'vocab_size': vocab_size, 'pad_idx': word_to_idx['<PAD>']}

        self.__dict__.update(params)

    @property
    def dict(self):
        return self.__dict__


def build_tokenizer():

    data_dir = Path().cwd() / 'data'
    # corpus_file = os.path.join(data_dir, 'ratings.txt')
    corpus_file = os.path.join(data_dir, 'ratings_train.txt')

    df = pd.read_table(corpus_file, encoding='utf-8')
    # df = df.head(2000)

    word_extractor = WordExtractor()
    word_extractor.train(str(df['document']))
    words = word_extractor.extract()
    cohesion_score = {word:score.cohesion_forward for word, score in words.items()}
    tokenizer = MaxScoreTokenizer(scores=cohesion_score)

    with open('pickles/tokenizer.pickle', 'wb') as pickle_out:
        pickle.dump(tokenizer, pickle_out)


def padding_sentence(params):

    max_sequence_length = params.max_sequence_length
    mode = params.mode

    pickle_tokenizer = open('pickles/tokenizer.pickle', 'rb')
    file_vocab = open('pickles/vocab.pickle', 'rb') # word_to_idx
    vocab = pickle.load(file_vocab)
    tokenizer = pickle.load(pickle_tokenizer)

    data_dir = Path().cwd() / 'data'

    corpus_file = os.path.join(data_dir, f'ratings_{mode}.txt')

    df = pd.read_table(corpus_file, encoding='utf-8')
    # df = df.head(200)

    label = torch.LongTensor(df['label'])

    df_length = len(df)

    sentence_list = []
    for _, row in tqdm(df.iterrows()):
        sentence_list.append(tokenizer.tokenize(str(row[1])))

    input_sentence = []
    for i, row in enumerate(sentence_list):
        temp_list = []
        row_length = len(row)
        for word in row:
            temp_list.append(vocab[word])
        if row_length < max_sequence_length:
            for _ in range(max_sequence_length - row_length):
                temp_list.append(vocab['<PAD>'])
        input_sentence.append(temp_list)


    # to Tensor
    input_sentence = np.array(input_sentence).reshape(-1, )
    input_sentence = torch.LongTensor(input_sentence).view(df_length, -1)

    return input_sentence, label


def make_iter(params, inputs, labels):

    data = torch.utils.data.TensorDataset(inputs, labels)

    if params.mode == 'train':
        shuffle = True
    else:
        shuffle = False

    data_loader = torch.utils.data.DataLoader(data, batch_size=params.batch_size, shuffle=shuffle)

    return data_loader