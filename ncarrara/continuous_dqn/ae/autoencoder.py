# -*- coding: utf-8 -*
import torch
import torch.utils.data
import numpy as np
from torch import nn
import torch.nn.functional as F
import logging

from ncarrara.continuous_dqn.tools.configuration import C

logger = logging.getLogger(__name__)


class Autoencoder(nn.Module):
    def __init__(self, n, min_n, max_n,device):
        """n_coding must be a power 2 number"""
        super(Autoencoder, self).__init__()

        encoder_layers = []
        encoder_layers.append(nn.Linear(n, max_n))
        encoder_layers.append(nn.ReLU(True))
        this_n = max_n
        while this_n > min_n:
            encoder_layers.append(nn.Linear(this_n, int(this_n / 2)))
            encoder_layers.append(nn.ReLU(True))
            this_n = int(this_n / 2)

        decoder_layers = []
        this_n = min_n
        while this_n < max_n:
            decoder_layers.append(nn.Linear(this_n, int(this_n * 2)))
            decoder_layers.append(nn.ReLU(True))
            this_n = int(this_n * 2)
        decoder_layers.append(nn.Linear(max_n, n))
        decoder_layers.append(nn.Tanh())

        self.encoder = nn.Sequential(*encoder_layers)
        self.decoder = nn.Sequential(*decoder_layers)

        self.to(C.device)
        self.std = None
        self.mean = None
        self.to(device)
        # print(self)

    def set_normalization_params(self, mean, std):
        std[std == 0.] = 1.
        self.std = std
        self.mean = mean

    def forward(self, x):
        if self.std is not None and self.mean is not None:
            # print(x.shape)
            # print(self.mean.shape)
            x = (x.float() - self.mean.float()) / self.std.float()
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def fit(self, datas, weight_decay=1e-5, n_epochs=10, stop_loss=0.01, size_minibatch=None, optimizer=None,
            loss_function=None, normalize=False):
        # print(datas.shape)
        means = torch.mean(datas, dim=0)
        stds = torch.std(datas, dim=0)
        # print(means.shape)
        # print(means)
        # exit()
        # exit()
        if normalize:
            self.set_normalization_params(means, stds)
        # if size_minibatch is None:
        #     size_minibatch = datas.shape[0]
        # if optimizer is None:
        #     optimizer = torch.optim.Adam(self.parameters(), lr=0.001, weight_decay=weight_decay)
        # if loss_function is None:
        #     loss_function = F.mse_loss
        logger.info("[fit] fitting ...")
        for epoch in range(n_epochs):
            random_indexes = torch.LongTensor(np.random.choice(range(datas.shape[0]), size_minibatch)).to(C.device)
            mini_batch = datas.index_select(0, random_indexes).to(C.device)
            loss = loss_function(self(mini_batch), mini_batch)
            optimizer.zero_grad()
            loss.backward()
            for param in self.parameters():
                param.grad.data.clamp_(-1, 1)
            optimizer.step()
            if epoch % (n_epochs / 10) == 0:
                logger.info('[fit] epoch [{}/{}], loss:{:.4f}'.format(epoch, n_epochs - 1, loss.data))
            if loss.data < stop_loss:
                break
        logger.info('[fit] final loss:{:.4f}'.format(loss.data))
        logger.info('[fit] fitting ... Done')
