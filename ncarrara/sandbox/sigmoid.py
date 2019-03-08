xxx = [torch.sigmoid(100. * torch.tensor(x)) for x in np.linspace(-4, 4, 50)]
import matplotlib.pyplot as plt

plt.plot(range(len(xxx)), xxx)
plt.savefig(self.workspace + "/sigmoid")
plt.close()
