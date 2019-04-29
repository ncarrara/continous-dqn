# -*- coding: utf-8 -*
from collections import namedtuple
import torch
import torch.utils.data
import numpy as np
from torch import nn
import torch.nn.functional as F
import logging
from ncarrara.continuous_dqn.tools.configuration import C

logger = logging.getLogger(__name__)
BatchAE = namedtuple("BatchAE", ("X", "A"))


class AutoEncoder(nn.Module):
    def __init__(self, n_in, n_out, min_n, max_n, device, type_ae="AE",reset_type="XAVIER", N_actions=None):
        """n_coding must be a power 2 number"""
        super(AutoEncoder, self).__init__()
        self.reset_type=reset_type
        self.n_in = n_in
        self.N_actions = N_actions
        self.type_ae = type_ae
        encoder_layers = [nn.Linear(n_in, max_n), nn.ReLU()]
        this_n = max_n
        while this_n > min_n:
            encoder_layers.append(nn.Linear(this_n, int(this_n / 2)))
            encoder_layers.append(nn.ReLU())
            this_n = int(this_n / 2)

        decoder_layers = []
        this_n = min_n
        while this_n < max_n:
            decoder_layers.append(nn.Linear(this_n, int(this_n * 2)))
            decoder_layers.append(nn.ReLU())
            this_n = int(this_n * 2)
        decoder_layers.append(nn.Linear(max_n, n_out))
        # decoder_layers.append(nn.Tanh())

        self.encoder = nn.Sequential(*encoder_layers)
        self.decoder = nn.Sequential(*decoder_layers)

        self.to(C.device)
        self.std = None
        self.mean = None
        self.to(device)
        print(self)
        self.reset()


    def reset(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if hasattr(m, 'weight'):
            if self.reset_type == "XAVIER":
                torch.nn.init.xavier_uniform_(m.weight.data)
            elif self.reset_type == "ZEROS":
                torch.nn.init.constant_(m.weight.data, 0.)
            else:
                raise ValueError("Unknown reset type")
        if hasattr(m, 'bias'):
            torch.nn.init.constant_(m.bias.data, 0.)

    def set_normalization_params(self, mean, std):
        std[std == 0.] = 1.
        self.std = std
        self.mean = mean

    def forward(self, x):
        # if self.std is not None and self.mean is not None:
        #     # print(x.shape)
        #     # print(self.mean.shape)
        #     x = (x.float() - self.mean.float()) / self.std.float()
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def fit(self, datas, n_epochs=10, stop_loss=0.01, size_minibatch=None,optimizer=None,
            loss_function=None, normalize=False,writer=None):

        # means = torch.mean(datas.X, dim=0)
        # stds = torch.std(datas.X, dim=0)
        # if normalize:
        #     self.set_normalization_params(means, stds)
        if size_minibatch is None:
            size_minibatch = datas.X.shape[0]
        if optimizer is None:
            optimizer = torch.optim.Adam(self.parameters(), lr=0.001, weight_decay=0.0)
        if loss_function is None:
            loss_function = F.mse_loss
        logger.info("[fit] fitting ...")
        for epoch in range(n_epochs):
            random_indexes = torch.LongTensor(np.random.choice(range(datas.X.shape[0]), size_minibatch)).to(C.device)
            X = datas.X.index_select(0, random_indexes).to(C.device)
            Y_ = self(X)
            Y = Y_.gather(1, datas.A)

            loss = loss_function(Y, X)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if writer is not None:
                writer.add_scalar('loss/episode', loss.data, epoch)
            if epoch % (n_epochs / 10) == 0:
                # for x, y in zip(X, Y):
                    # diff = torch.abs(x- y)
                    # print("-----")
                    # rounded = torch.round(diff * 10 ** n_digits) / (10 ** n_digits)
                    # for x,y,z in zip(x.cpu().detach().numpy(),y.cpu().detach().numpy(),diff.cpu().detach().numpy()):
                    #     print("x={:.2f} y={:.2f} -> diff={:.2f}".format(x,y,z))
                logger.info('[fit] epoch [{}/{}], loss:{:.4f}'.format(epoch, n_epochs - 1, loss.data))
            if loss.data < stop_loss:
                break
        logger.info('[fit] final loss:{:.4f}'.format(loss.data))
        logger.info('[fit] fitting ... Done')
