"""Neural Turing Machine

arXiv:1410.5401v2
"""
import functools
import itertools
import numpy as np
import torch
import torch.nn.functional as F

from copy import deepcopy
from datasets import MyDataset, get_rand_vector_sequence
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import MyModuleList


class FeedforwardController(torch.nn.Module):
    def __init__(self,
                 data_dim,
                 output_dim,
                 memory_size,
                 nread_heads=1,
                 nwrite_heads=1):
        super().__init__()
        self.memory_size = memory_size
        self.nread_heads = nread_heads
        self.nwrite_heads = nwrite_heads
        self.preprocess_out_dim = 64
        self.preprocess = MyModuleList([
            (torch.nn.Linear(data_dim, 128), F.leaky_relu),
            (torch.nn.Linear(128, self.preprocess_out_dim), F.leaky_relu),
        ])

        self.rheads = [self.make_rw_head() for _ in range(nread_heads)]
        self.wheads = [self.make_rw_head() for _ in range(nwrite_heads)]
        for head in self.wheads:
            head.update({
                'erase':
                torch.nn.Linear(self.preprocess_out_dim, memory_size[1]),
                'add':
                torch.nn.Linear(self.preprocess_out_dim, memory_size[1]),
            })
        softmax = functools.partial(F.softmax, dim=1)
        self.rw_activations = {
            'key': F.leaky_relu,
            'key_strength': F.leaky_relu,
            'interpolation_gate': torch.sigmoid,
            'shift_weighting': softmax,
            'weight_sharpen': lambda x: F.relu(x) + 1,
            'erase': torch.sigmoid,
            'add': torch.sigmoid,
        }

        input_dim = data_dim + nread_heads * np.prod(memory_size)
        self.out_head = MyModuleList([
            (torch.nn.Linear(input_dim, 256), F.leaky_relu),
            (torch.nn.Linear(256, 128), F.leaky_relu),
            (torch.nn.Linear(128, output_dim), torch.sigmoid),
        ])

    def forward(self, x, memory):
        pp_out = self.preprocess.compose_modules()(x)
        routs = []
        for head in self.rheads:
            routs.append({
                k: self.rw_activations[k](layer(pp_out))
                for k, layer in head.items()
            })
        wouts = []
        for head in self.wheads:
            wouts.append({
                k: self.rw_activations[k](layer(pp_out))
                for k, layer in head.items()
            })
        rweights = [
            self.get_weight(out, prev_weight, memory)
            for out, prev_weight in zip(routs, self.prev_rweights)
        ]
        wweights = [
            self.get_weight(out, prev_weight, memory)
            for out, prev_weight in zip(wouts, self.prev_wweights)
        ]
        mem_read = torch.stack(
            [weight.t() * memory for weight in rweights], dim=-1)
        aug_x = torch.cat([x, mem_read.reshape(1, -1)], dim=1)
        yhat = self.out_head.compose_modules()(aug_x)
        self.prev_rweights = rweights
        self.prev_wweights = wweights
        return yhat, self.write(wweights, wouts, memory)

    def make_rw_head(self):
        return torch.nn.ModuleDict({
            'key':
            torch.nn.Linear(self.preprocess_out_dim, self.memory_size[1]),
            'key_strength':
            torch.nn.Linear(self.preprocess_out_dim, 1),
            'interpolation_gate':
            torch.nn.Linear(self.preprocess_out_dim, 1),
            'shift_weighting':
            torch.nn.Linear(self.preprocess_out_dim, self.memory_size[0]),
            'weight_sharpen':
            torch.nn.Linear(self.preprocess_out_dim, 1),
        })

    def reset_rw_weights(self):
        self.prev_rweights = [
            .01 * torch.randn(self.memory_size[0])
            for _ in range(self.nread_heads)
        ]
        self.prev_wweights = [
            .01 * torch.randn(self.memory_size[0])
            for _ in range(self.nwrite_heads)
        ]

    @staticmethod
    def get_weight(out, prev_weight, memory):
        key_rep = out['key'].repeat(memory.shape[0], 1)
        norm_weight = F.softmax(
            out['key_strength'] * F.cosine_similarity(key_rep, memory), dim=1)
        gated_weight = out['interpolation_gate'] * norm_weight + \
            (1 - out['interpolation_gate']) * prev_weight
        shifted_weight = torch.empty_like(gated_weight)
        idx = torch.tensor(range(memory.shape[0]), dtype=torch.long)
        for i in range(memory.shape[0]):
            shifted_weight[..., i] = torch.dot(
                gated_weight.squeeze(),
                out['shift_weighting'].squeeze()[i - idx])
        return F.softmax(shifted_weight**out['weight_sharpen'], dim=1)

    @staticmethod
    def write(weights, wouts, memory):
        # TODO maybe store state to avoid allocating a new tensor each call
        next_mem = torch.empty_like(memory)
        next_mem.copy_(memory)
        erases = [out['erase'] for out in wouts]
        adds = [out['add'] for out in wouts]
        for w, e in zip(weights, erases):
            e_rep = e.repeat(memory.shape[0], 1)
            next_mem *= torch.ones_like(e) - w.t() * e_rep
        for w, a in zip(weights, adds):
            a_rep = a.repeat(memory.shape[0], 1)
            next_mem += w.t() * a_rep
        return next_mem

    def reset(self):
        self.reset_rw_weights()
        for m in self.preprocess:
            m.reset_parameters()
        for head in itertools.chain(self.rheads, self.wheads):
            for m in head.values():
                m.reset_parameters()


class RecurrentController(torch.nn.Module):
    pass


class NeuralTuringMachine:
    def __init__(self,
                 data_dim,
                 output_dim,
                 memory_size,
                 learning_rate=1e-4,
                 nread_heads=1,
                 nwrite_heads=1,
                 controller=None,
                 device=None):
        assert len(memory_size) == 2, 'Memory is 2-dimensional'
        if device is None or str(device) != 'cuda':
            self.dev = torch.device('cpu')
        else:
            self.dev = device
        self.memory = None
        self.memory_size = memory_size
        self.nread_heads = nread_heads
        self.nwrite_heads = nwrite_heads
        if controller is None:
            controller = FeedforwardController
        self.controller = controller(
            data_dim,
            output_dim,
            memory_size,
            nread_heads=nread_heads,
            nwrite_heads=nwrite_heads)
        self.optimizer = torch.optim.Adam(
            self.controller.parameters(), lr=learning_rate)

    def fit(self, dataset, epochs=32, verbose=False):
        loader = DataLoader(dataset)
        for epoch in range(epochs):
            cum_loss = 0.
            if verbose:
                loader = tqdm(loader)
            for x, y in loader:
                self.memory = self.init_memory()
                self.controller.reset()

                x, y = x.squeeze(0).to(self.dev), y.squeeze(0).to(self.dev)
                yhat = torch.empty_like(y)
                for i, letter in enumerate(x):
                    yhat[i], self.memory = self.controller(letter, self.memory)
                loss = F.binary_cross_entropy_with_logits(
                    yhat, y, reduction='sum')
                cum_loss += loss.item()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            if verbose:
                print('epoch', epoch, cum_loss)

    def init_memory(self):
        return .01 * torch.randn(*self.memory_size)

    @torch.no_grad()
    def predict(self, x):
        yhat, self.memory = self.controller(x, self.memory)
        return yhat


if __name__ == '__main__':
    if torch.cuda.is_available():
        dev = torch.device('cuda')
    else:
        dev = torch.device('cpu')

    # Copy task from section 4.1
    vsize = 8
    x_train = get_rand_vector_sequence(1, 21, vsize=vsize, num_samples=100)
    y_train = deepcopy(x_train)
    dataset = MyDataset(x_train, y_train)
    mem_size = (100, 20)
    ntm = NeuralTuringMachine(
        vsize, vsize, mem_size, controller=FeedforwardController, device=dev)
    ntm.fit(dataset, verbose=True)
