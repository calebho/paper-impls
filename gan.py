"""Generative Adversarial Network

https://arxiv.org/abs/1406.2661v1
"""
import argparse
import functools
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from datasets import MyDataset, get_mnist
from torch.utils.data import DataLoader
from utils import MyModuleList


class Discriminator(torch.nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        dropout = functools.partial(F.dropout, training=True, p=.3)
        self.module_list = MyModuleList([
            (torch.nn.Linear(input_dim, 1024), F.leaky_relu, dropout),
            (torch.nn.Linear(1024, 512), F.leaky_relu, dropout),
            (torch.nn.Linear(512, 256), F.leaky_relu, dropout),
            (torch.nn.Linear(256, 1), F.sigmoid),
        ])

    def forward(self, x):
        return self.module_list.compose_modules()(x)


class Generator(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.module_list = MyModuleList([
            (torch.nn.Linear(input_dim, 256), F.leaky_relu),
            (torch.nn.Linear(256, 512), F.leaky_relu),
            (torch.nn.Linear(512, 1024), F.leaky_relu),
            (torch.nn.Linear(1024, output_dim), F.sigmoid),
        ])

    def forward(self, x):
        return self.module_list.compose_modules()(x)


def discriminator_loss(y_pred, y_gen):
    label_pred = torch.ones_like(y_pred)
    label_gen = torch.zeros_like(y_gen)
    bce = torch.nn.BCELoss()
    pred_loss = bce(y_pred, label_pred)
    gen_loss = bce(y_gen, label_gen)
    return pred_loss + gen_loss


def generator_loss(y_gen):
    bce = torch.nn.BCELoss()
    labels = torch.ones_like(y_gen)
    return bce(y_gen, labels)


class GAN:
    def __init__(self,
                 data_dim,
                 gen_input_dim=256,
                 disc_learning_rate=1e-4,
                 gen_learning_rate=1e-4,
                 device=None):
        self.disc_learning_rate = disc_learning_rate
        self.gen_learning_rate = gen_learning_rate

        self.data_dim = data_dim
        self.gen_input_dim = gen_input_dim
        self.disc = Discriminator(data_dim)
        self.gen = Generator(self.gen_input_dim, data_dim)
        if device is None or str(device) != 'cuda':
            self.dev = torch.device('cpu')
        else:
            self.dev = device
            self.disc = self.disc.cuda()
            self.gen = self.gen.cuda()

        self.disc_optimizer = torch.optim.Adam(
            self.disc.parameters(), lr=self.disc_learning_rate)
        self.gen_optimizer = torch.optim.Adam(
            self.gen.parameters(), lr=self.gen_learning_rate)

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
            for x in loader:
                x = x.to(self.dev)

                # train discriminator
                self.disc.train()
                for _ in range(discriminator_iters):
                    noise = torch.randn(
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

                # train generator
                self.disc.eval()
                noise = torch.randn(
                    batch_size, self.gen_input_dim, device=self.dev)
                y_gen = self.disc(self.gen(noise))
                gen_loss = generator_loss(y_gen)
                gen_cum_loss += gen_loss.item()
                self.gen_optimizer.zero_grad()
                gen_loss.backward()
                self.gen_optimizer.step()
            if verbose:
                print('epoch', epoch, disc_cum_loss, gen_cum_loss)

    def predict(self, x):
        return self.disc.eval()(x)

    def evaluate(self, x, y):
        yhat = self.predict(x).cpu().detach().numpy()
        y = y.detach().numpy()
        yhat[yhat[:, 0] > .5] = 1
        yhat[yhat[:, 0] <= .5] = 0
        return np.mean(y == yhat[:, 0])

    def generate(self, noise):
        return self.gen(noise)


def make_mask(ys, labels):
    mask = np.zeros_like(ys, dtype=bool)
    for label in labels:
        mask |= ys == label
    return mask


if __name__ == '__main__':
    p = argparse.ArgumentParser(
        description='Fit a GAN to MNIST',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument(
        '-e',
        '--epochs',
        type=int,
        default=32,
        help='Number of training epochs.')
    p.add_argument(
        '-b',
        '--batch-size',
        type=int,
        default=256,
        help='Training batch size.')
    p.add_argument(
        '--gen-learning-rate',
        type=float,
        default=1e-4,
        help='Generator learning rate.')
    p.add_argument(
        '--disc-learning-rate',
        type=float,
        default=1e-4,
        help='Discriminator learning rate.')
    p.add_argument(
        '-g',
        '--gen-input-dim',
        type=int,
        default=100,
        help='Generator input dimension.')
    p.add_argument(
        '--num-gen',
        type=int,
        default=0,
        help='Number of digits to generate after training.')
    p.add_argument(
        '-v',
        '--verbose',
        action='store_true',
        help='Print losses after each epoch')
    args = p.parse_args()

    train_x, train_y, test_x, test_y = get_mnist()
    # normalize inputs to [0, 1]
    train_x /= 255
    test_x /= 255

    if torch.cuda.is_available():
        dev = torch.device('cuda')
    else:
        dev = torch.device('cpu')

    numbers = range(10)  # subset of numbers to train on
    train_mask = make_mask(train_y, numbers)
    xs = train_x[train_mask]
    dataset = MyDataset(xs)
    m = GAN(
        28 * 28,
        gen_input_dim=args.gen_input_dim,
        disc_learning_rate=args.disc_learning_rate,
        gen_learning_rate=args.gen_learning_rate,
        device=dev)
    m.fit(
        dataset,
        batch_size=args.batch_size,
        epochs=args.epochs,
        verbose=args.verbose)

    # generate digits and save
    if args.num_gen:
        noise = torch.randn(args.num_gen, args.gen_input_dim, device=dev)
        y = np.reshape(
            m.generate(noise).cpu().detach().numpy(), (args.num_gen, 28, 28))
        for i in range(args.num_gen):
            plt.imsave(
                'output/gan/im{:02d}.png'.format(i),
                y[i],
                cmap='gray',
                format='png')
