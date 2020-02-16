import time

import torch

from model import layers
from model.cnn import SentenceCNN

class Trainer:
    def __init__(self, params, mode):
        self.params = params
        self.num_classes = params.num_classes
        self.vocab_size = params.
        # Train mode

        self.model = SentenceCNN()