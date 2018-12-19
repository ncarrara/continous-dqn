# -*- coding: utf-8 -*
import torch
import torch.utils.data
import numpy as np
from torch import nn
import torch.nn.functional as F
import logging
import configuration as config

logger = logging.getLogger(__name__)


class Autoencoder(nn.Module):
    def __init__(self, n, min_n, max_n):
        """n_coding must be a power 2 number"""
        super(Autoencoder, self).__init__()

        encoder_layers = []
        encoder_layers.append(nn.Linear(n, max_n))
        encoder_layers.append(nn.ReLU(True))
        this_n = max_n
        while this_n > min_n:
            encoder_layers.append(nn.Linear(this_n, int(this_n / 2)))
            encoder_layers.append(nn.ReLU(True))
            this_n = int(this_n/2)

        decoder_layers = []
        this_n = min_n
        while this_n < max_n:
            decoder_layers.append(nn.Linear(this_n, int(this_n * 2)))
            decoder_layers.append(nn.ReLU(True))
            this_n = int(this_n* 2)
        decoder_layers.append(nn.Linear(max_n, n))
        decoder_layers.append(nn.Tanh())
        # decoder_layers.append(nn.Sigmoid())

        # print(encoder_layers)
        # print(decoder_layers)


        self.encoder = nn.Sequential(*encoder_layers)
        self.decoder = nn.Sequential(*decoder_layers)

        # self.encoder = nn.Sequential(
        #     nn.Linear(n, 64), nn.ReLU(True),
        #     nn.Linear(64, 32), nn.ReLU(True),
        #     nn.Linear(32, 16), nn.ReLU(True),
        #     nn.Linear(16, 8))
        # self.decoder = nn.Sequential(
        #     nn.Linear(8, 16), nn.ReLU(True),
        #     nn.Linear(16, 32), nn.ReLU(True),
        #     nn.Linear(32, 64), nn.ReLU(True),
        #     nn.Linear(64, n), nn.Tanh())

        self.to(config.DEVICE)
        self.std = None
        self.mean = None

    def set_normalization_params(self, mean, std):
        # if std is not Net.DONT_NORMALIZE_YET:
        std[std == 0.] = 1.  # on s'en moque, on divisera 0 par 1
        self.std = std
        self.mean = mean

    def forward(self, x):
        if self.std is not None and self.mean is not None:
            x = (x.float() - self.mean.float()) / self.std.float()
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def fit(self, datas, weight_decay=1e-5, n_epochs=10, stop_loss=0.01, size_minibatch=None, optimizer=None,
            criterion=None, normalize=False):
        means = torch.mean(datas, dim=0)
        stds = torch.std(datas, dim=0)
        if normalize:
            self.set_normalization_params(means, stds)
        if size_minibatch is None:
            size_minibatch = datas.shape[0]
        if optimizer is None:
            optimizer = torch.optim.Adam(self.parameters(), lr=0.001, weight_decay=weight_decay)
        if criterion is None:
            criterion = F.mse_loss
        logger.info("[fit] fitting ...")

        # creating minibatch
        # newdim = int(datas.shape[0] / size_minibatch)
        # reste = datas.shape[0] % size_minibatch
        # minibatchs = datas[:-reste].reshape((newdim, -1, datas.shape[1]))
        # lastbatch = None
        # if reste >0:
        #     lastbatch = datas[-reste:]
        #     lastbatch = lastbatch.reshape((lastbatch.shape[0], -1, lastbatch.shape[1]))

        # trainloader = torch.utils.data.DataLoader(datas, batch_size=32, shuffle=True, num_workers=8)

        for epoch in range(n_epochs):
            random_indexes = torch.LongTensor(np.random.choice(range(datas.shape[0]), size_minibatch)).to(config.DEVICE)
            mini_batch = datas.index_select(0, random_indexes).to(config.DEVICE)
            loss = criterion(self(mini_batch), mini_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # randomminibatch
            # for i, minibatch in enumerate(trainloader, 0):
            #     # print(i,data)
            #     loss = self.fit_minibatch(minibatch.cuda(), optimizer)
            #     print(i,loss.data)

            # mini batch GD
            # for minibatch in minibatchs:
            #     loss = self.fit_minibatch(minibatch, optimizer)
            # if lastbatch is not None:
            #     loss = self.fit_minibatch(lastbatch, optimizer)
            # SGD
            # for data in datas:
            #     y = data
            #     y_ = self(y)
            #     loss = self.criterion(y, y_)
            #     print(loss.data)
            #     optimizer.zero_grad()
            #     loss.backward()
            #     optimizer.step()
            # if loss.data < stop_loss:
            #     break

            #  FULL GD
            # loss = self.criterion(datas, self(datas))
            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()
            if epoch % (n_epochs / 10) == 0:
                logger.info('[fit] epoch [{}/{}], loss:{:.4f}'.format(epoch, n_epochs - 1, loss.data))
            if loss.data < stop_loss:
                break
        logger.info('[fit] final loss:{:.4f}'.format(loss.data))
        logger.info('[fit] fitting ... Done')
