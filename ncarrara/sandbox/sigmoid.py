import torch
import numpy as np
X= np.linspace(-10, 10, 50)
xxx = [torch.sigmoid(10. * torch.tensor(x)) for x in X]
import matplotlib.pyplot as plt

plt.plot(X, xxx)
plt.savefig("sigmoid")
plt.close()
