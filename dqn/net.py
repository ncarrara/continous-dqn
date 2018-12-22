import torch
import torch.nn.functional as F


class Net(torch.nn.Module):
    DONT_NORMALIZE_YET = None

    def _init_weights(self, m):
        if hasattr(m, 'weight'):
            torch.nn.init.xavier_uniform_(m.weight.data)
            # torch.nn.init.constant_(m.weight.data, 0.)

    def __init__(self, sizes, activation=F.relu, normalize=False):
        super(Net, self).__init__()
        self.normalize = normalize
        self.activation = activation
        self.layers = []
        for i in range(0, len(sizes) - 2):
            module = torch.nn.Linear(sizes[i], sizes[i + 1])
            self.layers.append(module)
            self.add_module("h_" + str(i), module)
        self.predict = torch.nn.Linear(sizes[-2], sizes[-1])
        self.std = Net.DONT_NORMALIZE_YET
        self.mean = Net.DONT_NORMALIZE_YET

    def set_normalization_params(self, mean, std):
        if std is not Net.DONT_NORMALIZE_YET:
            std[std == 0.] = 1.  # on s'en moque, on divisera 0 par 1.
        self.std = std
        self.mean = mean

    def forward(self, x):
        # print "x : ",x
        if self.normalize and self.mean is not Net.DONT_NORMALIZE_YET and self.std is not Net.DONT_NORMALIZE_YET:  # hasattr(self, "normalize"):
            x = (x.float() - self.mean.float()) / self.std.float()
        for layer in self.layers:
            x = self.activation(layer(x))
        x = self.predict(x)  # linear output
        return x.view(x.size(0), -1)

    def reset(self):
        self.apply(self._init_weights)
