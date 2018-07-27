"""Variational Auto-Encoder

arXiv:1312.6114v10
"""
import torch
import torch.nn.functional as F

from datasets import MyDataset, get_mnist
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.utils.data import DataLoader


class AutoEncoder(torch.nn.Module):
    """Auto-Encoder with Gaussian output"""

    def __init__(self, input_dim, latent_dim=2, num_noise_samples=1):
        super().__init__()
        self.num_noise_samples = num_noise_samples
        self.latent_dim = latent_dim
        self.enc_fc1 = torch.nn.Linear(input_dim, 512)
        self.enc_fc2 = torch.nn.Linear(512, latent_dim)
        self.enc_fc3 = torch.nn.Linear(512, latent_dim)

        self.dec_fc1 = torch.nn.Linear(latent_dim, 512)
        self.dec_fc2 = torch.nn.Linear(512, input_dim)
        self.dec_fc3 = torch.nn.Linear(512, input_dim)

    def forward(self, x):
        # encode
        x = F.tanh(self.enc_fc1(x))
        enc_mu = self.enc_fc2(x)
        enc_log_sigma2 = self.enc_fc3(x)

        # noise for computing latent variable z
        eps = torch.randn(
            self.num_noise_samples, self.latent_dim, device=x.device)

        # decode
        x = F.tanh(self.dec_fc1(enc_mu + enc_log_sigma2 * eps))
        return self.dec_fc2(x), self.dec_fc3(x)  # mu and log(sigma^2)


def elbo(mu, sigma2, log_prob):
    """Evidence lower bound (ELBO) for Gaussian prior and posterior"""
    neg_kl_div = 5 * torch.mean(1 + torch.log(sigma2) - mu**2 - sigma2, 1)
    neg_reconstruction_err = torch.mean(log_prob, 1)
    return torch.mean(neg_kl_div + neg_reconstruction_err)


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

    def fit(self, dataset, epochs=40, batch_size=64, verbose=False):
        loader = DataLoader(dataset, batch_size)
        for epoch in range(epochs):
            cum_loss = 0.
            for x in loader:
                x = x.to(self.dev)
                mus, log_sigma2s = self.ae(x)
                log_probs = torch.empty((x.shape[0], 1), device=self.dev)
                for i, (x_, mu, log_sigma2) in enumerate(
                        zip(x, mus, log_sigma2s)):
                    mv_norm = MultivariateNormal(
                        mu,
                        torch.exp(log_sigma2) * torch.eye(
                            x.shape[1], device=self.dev))
                    log_probs[i] = mv_norm.log_prob(x_)
                loss = -elbo(mus, torch.exp(log_sigma2s), log_probs)
                cum_loss += loss.item()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            if verbose:
                print('epoch', epoch, cum_loss)


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
    m.fit(dataset, verbose=True)
