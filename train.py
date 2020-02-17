import time

import torch.nn as nn
import torch.optim as optim

from model.cnn import SentenceCNN

class Trainer:
    def __init__(self, params):
        self.params = params

        # Train mode
        self.model = SentenceCNN(self.params.num_classes, self.params.vocab_size, self.params.embedding_dim)
        self.model.to(self.params.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=params.learning_rate)
        self.loss_fn = nn.CrossEntropyLoss()
        self.loss_fn.to(self.params.device)

    def train(self, data_loader):

        start = time.time()
        check = 100
        loss_list = []

        for epoch in range(self.params.epoch):
            loss_sum = 0.0

            for i, data in enumerate(data_loader):

                inputs, labels = data[0].to(self.params.device), data[1].to(self.params.device)

                self.optimizer.zero_grad()

                outputs = self.model(inputs)

                loss = self.loss_fn(outputs, labels)
                loss.backward()
                self.optimizer.step()

                loss_sum += loss.item()

                if i % check == 0 and i != 0:
                    elap = int(time.time() - start)
                    loss_list.append(loss_sum/check)
                    elapsed = (elap//3600, (elap % 3600) // 60, str(int((elap % 3600) % 60)))
                    print(f'Epoch: {epoch}, Iteration: {i}, Loss: {loss_sum/check:.2f}, '
                          f'Elapsed time: {elapsed[0]:.0f}h {elapsed[1]:.0f}m {elapsed[2]}s')
                    loss_sum = 0.0

    # def evaluate(self):
    #     self.model.eval()
