import torch

from ncarrara.utils.torch_utils import BaseModule
import numpy as np


class TNN(BaseModule):
    def __init__(self, n_in, n_out, intra_layers, activation_type="RELU", reset_type="XAVIER", normalize=None):
        super(TNN, self).__init__(activation_type, reset_type, normalize)
        all_layers = [n_in] + intra_layers + [n_out]
        self.layers = []
        for i in range(0, len(all_layers) - 2):
            module = torch.nn.Linear(all_layers[i], all_layers[i + 1])
            self.layers.append(module)
            self.add_module("h_" + str(i), module)
        self.predict = torch.nn.Linear(all_layers[-2], all_layers[-1])

        self.concat_layer = torch.nn.Linear(n_out * 2 + 1, all_layers[-1])

    def set_Q_source(self, Q_source, ae_score):
        self.Q_source = Q_source
        self.ae_score = torch.tensor(ae_score)

    def reset(self):
        self.Q_source = None
        self.ae_score = np.nan
        super(TNN, self).reset()

    def forward(self, s):
        input_Q = s
        if self.normalize:
            input_Q = (input_Q.float() - self.mean.float()) / self.std.float()

        for layer in self.layers:
            input_Q = self.activation(layer(input_Q))
        out_Q = self.predict(input_Q)
        out_Q_transfer = self.Q_source(s)

        in_concat = torch.cat(out_Q, out_Q_transfer, self.ae_score, dim=1)

        y = self.concat_layer(in_concat)

        return y.view(y.size(0), -1)
