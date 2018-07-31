"""Variational Auto-Encoder

arXiv:1312.6114v10
"""
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from datasets import MyDataset, get_mnist
from functools import reduce
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import compose


class AutoEncoder(torch.nn.Module):
    """Auto-Encoder with Bernoulli output"""

    def __init__(self, input_dim, latent_dim=2, num_noise_samples=1):
        super().__init__()
        self.num_noise_samples = num_noise_samples
        self.latent_dim = latent_dim
        self.enc_fc1 = torch.nn.Linear(input_dim, 512)
        self.enc_fc2 = torch.nn.Linear(512, latent_dim)
        self.enc_fc3 = torch.nn.Linear(512, latent_dim)

        self.dec_fc1 = torch.nn.Linear(latent_dim, 512)
        self.dec_fc2 = torch.nn.Linear(512, input_dim)

    def encode(self, x):
        h = torch.tanh(self.enc_fc1(x))
        return self.enc_fc2(h), self.enc_fc3(h)

    def decode(self, mean, var):
        eps = torch.randn_like(var)
        f = reduce(compose,
                   [self.dec_fc1, torch.relu, self.dec_fc2, torch.sigmoid])
        return f(mean + var * eps)

    def forward(self, x):
        mean, logvar = self.encode(x)
        var = torch.exp(logvar)
        outs = [self.decode(mean, var) for _ in range(self.num_noise_samples)]
        return mean, var, torch.stack(outs, dim=1)


def neg_elbo(mu, sigma2, x, x_gen, weight=.5):
    """Negative evidence lower bound (ELBO) for Gaussian prior and Bernoulli
    posterior
    """
    kl_div = -.5 * torch.sum(1 + torch.log(sigma2) - mu**2 - sigma2)
    num_latent_samples = x_gen.shape[1]
    x = x.repeat(num_latent_samples, 1, 1).transpose(0, 1)
    reconstruction_err = F.binary_cross_entropy(x_gen, x, reduction='sum')
    return weight * kl_div + (1 - weight) * reconstruction_err


class VAE:
    def __init__(self,
                 data_dim,
                 latent_dim=2,
                 num_noise_samples=1,
                 learning_rate=1e-4,
                 device=None):
        self.data_dim = data_dim
        self.latent_dim = latent_dim
        self.num_noise_samples = num_noise_samples

        self.ae = AutoEncoder(data_dim, latent_dim, num_noise_samples)
        if device is None or str(device) != 'cuda':
            self.dev = torch.device('cpu')
        else:
            self.dev = device
            self.ae = self.ae.cuda()

        self.optimizer = torch.optim.Adam(
            self.ae.parameters(), lr=learning_rate)

    def fit(self,
            dataset,
            epochs=40,
            batch_size=64,
            loss_weight=.5,
            verbose=False):
        loader = DataLoader(dataset, batch_size)
        for epoch in range(epochs):
            cum_loss = 0.
            if verbose:
                loader = tqdm(loader)
            for x in loader:
                x = x.to(self.dev)
                mus, sigma2s, x_gen = self.ae(x)
                loss = len(dataset) / len(x) * neg_elbo(
                    mus, sigma2s, x, x_gen, weight=loss_weight)
                cum_loss += loss.item()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            if verbose:
                print('epoch', epoch, cum_loss)

    @torch.no_grad()
    def generate(self, mean, var):
        return self.ae.decode(mean, var)

    @torch.no_grad()
    def latent_representation(self, x):
        return self.ae.encode(x)


if __name__ == '__main__':
    train_x, train_y, test_x, test_y = get_mnist()
    train_x /= 255
    test_x /= 255

    if torch.cuda.is_available():
        dev = torch.device('cuda')
    else:
        dev = torch.device('cpu')

    dataset = MyDataset(train_x)
    m = VAE(28 * 28, device=dev)
    m.fit(dataset, epochs=5, verbose=True)
    mean = torch.randn(1, m.latent_dim)
    var = 10 * torch.rand(1, m.latent_dim)
    im = m.generate(mean, var)
    plt.imshow(im.detach().numpy().reshape(28, 28))
