"""Generative Adversarial Network

https://arxiv.org/abs/1406.2661v1
"""
import functools
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from datasets import MyDataset, get_mnist
from torch.utils.data import DataLoader
from utils import compose, interleave


class Discriminator(torch.nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.layers = torch.nn.ModuleList([
            torch.nn.Linear(input_dim, 128),
            torch.nn.Linear(128, 64),
            torch.nn.Linear(64, 2),
        ])
        self.activations = [
            F.relu,
            F.relu,
            functools.partial(F.log_softmax, dim=1),
        ]

    def forward(self, x):
        return functools.reduce(compose,
                                interleave(self.layers, self.activations))(x)


class Generator(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.layers = torch.nn.ModuleList([
            torch.nn.Linear(input_dim, 128),
            torch.nn.Linear(128, output_dim),
        ])
        self.activations = [
            F.relu,
            F.relu,
        ]

    def forward(self, x):
        return functools.reduce(compose,
                                interleave(self.layers, self.activations))(x)


def discriminator_loss(y_pred, y_gen):
    return -torch.mean(y_pred[:, 0] + torch.log(1 - torch.exp(y_gen[:, 0])))


def generator_loss(y_gen):
    return -torch.mean(y_gen[:, 0])


class GAN:
    def __init__(self, learning_rate=1e-4, device=None):
        self.learning_rate = learning_rate

        self.gen_input_dim = 256
        self.disc = Discriminator(28 * 28)
        self.gen = Generator(self.gen_input_dim, 28 * 28)
        if device is None or str(device) != 'cuda':
            self.dev = torch.device('cpu')
        else:
            self.dev = device
            self.disc = self.disc.cuda()
            self.gen = self.gen.cuda()

        self.disc_optimizer = torch.optim.Adam(
            self.disc.parameters(), lr=learning_rate)
        self.gen_optimizer = torch.optim.Adam(
            self.gen.parameters(), lr=learning_rate)

    def fit(self,
            dataset,
            epochs=40,
            batch_size=64,
            discriminator_iters=1,
            verbose=False):
        loader = DataLoader(dataset, batch_size)
        for epoch in range(epochs):
            disc_cum_loss = 0.
            gen_cum_loss = 0.
            for x, _ in loader:
                x = x.to(self.dev)
                for _ in range(discriminator_iters):
                    noise = torch.rand(
                        batch_size, self.gen_input_dim, device=self.dev)
                    x_gen = self.gen(noise)
                    y_pred = self.disc(x)
                    y_gen = self.disc(x_gen)
                    # if batch_size does not evenly divide number of train
                    # samples, then last batch will have fewer samples, causing
                    # y_pred and y_gen to have mismatched shapes
                    if y_gen.size()[0] != x.size()[0]:
                        y_gen = y_gen.narrow(0, 0, x.size()[0])
                    disc_loss = discriminator_loss(y_pred, y_gen)
                    disc_cum_loss += disc_loss.item()
                    self.disc_optimizer.zero_grad()
                    disc_loss.backward()
                    self.disc_optimizer.step()
                noise = torch.rand(
                    batch_size, self.gen_input_dim, device=self.dev)
                y_gen = self.disc(self.gen(noise))
                gen_loss = generator_loss(y_gen)
                gen_cum_loss += gen_loss.item()
                self.gen_optimizer.zero_grad()
                gen_loss.backward()
                self.gen_optimizer.step()
            if verbose:
                print('epoch', epoch, disc_cum_loss, gen_cum_loss)

    def predict(self, noise):
        return self.gen(noise).cpu().detach().numpy()


if __name__ == '__main__':
    train_x, train_y, test_x, test_y = get_mnist()
    # join training and test sets since we're not classifying
    train_x = np.concatenate([train_x, test_x])
    train_y = np.concatenate([train_y, test_y])

    if torch.cuda.is_available():
        dev = torch.device('cuda')
    else:
        dev = torch.device('cpu')

    # generate 7's
    number = 7
    xs = train_x[train_y == number]
    ys = np.reshape(train_y[train_y == number], (-1, 1))
    dataset = MyDataset(xs, ys)
    m = GAN(device=dev)
    m.fit(dataset, epochs=128, verbose=False)
    num_gen = 5
    noise = torch.rand(num_gen, 256, device=dev)
    y = np.reshape(m.predict(noise), (num_gen, 28, 28))
    for i in range(num_gen):
        plt.imsave(
            'output/gan/im{:02d}.png'.format(i),
            y[i],
            cmap='gray',
            format='png')
