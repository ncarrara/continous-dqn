import torch
from torch import nn
import logging

logger = logging.getLogger(__name__)


class Autoencoder(nn.Module):
    def __init__(self, n,criterion):
        super(Autoencoder, self).__init__()
        self.criterion=criterion
        self.encoder = nn.Sequential(
            nn.Linear(n, 16), nn.ReLU(True),
            # nn.Linear(64, 64), nn.ReLU(True),
            nn.Linear(16, 8), nn.ReLU(True),
            nn.Linear(8, 4))
        self.decoder = nn.Sequential(
            nn.Linear(4, 8), nn.ReLU(True),
            nn.Linear(8, 16), nn.ReLU(True),
            # nn.Linear(16, 128), nn.ReLU(True),
            nn.Linear(16, n), nn.Tanh())

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def fit(self, datas, learning_rate=0.001, weight_decay=1e-5, n_epochs=1000, stop_loss=0.01):
        logger.info("[fit] fitting ...")

        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
        for epoch in range(n_epochs):
            loss = self.criterion(datas,self(datas))
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
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if epoch%100==0:
                logger.info('[fit] epoch [{}/{}], loss:{:.4f}'.format(epoch, n_epochs-1, loss.data))
            if loss.data < stop_loss:
                break
        logger.info('[fit] final loss:{:.4f}'.format(loss.data))
        logger.info('[fit] fitting ... Done')
