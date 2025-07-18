""" The CNN model was implemented as published in the article https://nextjournal.com/gkoehler/pytorch-mnist """

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
from matplotlib import pyplot as plt
import seaborn as sns

figures_folder = "data/figures/cnn/"
os.makedirs(figures_folder, exist_ok=True)
input_folder = "data/input/cnn/"
os.makedirs(input_folder, exist_ok=True)

n_epochs = 10
batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.5
log_interval = 10

random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)


train_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('/files/', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=batch_size_train, shuffle=True)

test_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('/files/', train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=batch_size_test, shuffle=True)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)
    
network = Net()
optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)

train_losses = []
train_counter = []
test_losses = []
test_counter = [i*len(train_loader.dataset) for i in range(n_epochs + 1)]


def train(epoch, network, optimizer, train_loader):
  network.train()
  for batch_idx, (data, target) in enumerate(train_loader):
    optimizer.zero_grad()
    output = network(data)
    loss = F.nll_loss(output, target)
    loss.backward()
    optimizer.step()
    if batch_idx % log_interval == 0:
      print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        epoch, batch_idx * len(data), len(train_loader.dataset),
        100. * batch_idx / len(train_loader), loss.item()))
      train_losses.append(loss.item())
      train_counter.append(
        (batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))
      torch.save(network.state_dict(), input_folder+'cnn/model.pth')
      torch.save(optimizer.state_dict(), input_folder+'cnn/optimizer.pth')

def test(network, test_loader):
  network.eval()
  test_loss = 0
  correct = 0
  with torch.no_grad():
    for data, target in test_loader:
      output = network(data)
      test_loss += F.nll_loss(output, target, size_average=False).item()
      pred = output.data.max(1, keepdim=True)[1]
      correct += pred.eq(target.data.view_as(pred)).sum()
  test_loss /= len(test_loader.dataset)
  test_losses.append(test_loss)
  print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, len(test_loader.dataset),
    100. * correct / len(test_loader.dataset)))
  

test()
for epoch in range(1, n_epochs + 1):
  train(epoch)
  test()

# Set consistent global style
sns.set(style="whitegrid", context="paper")  # or "talk" for presentations

plt.rcParams.update({
    "figure.figsize": (8, 6),           # Width, height in inches
    "figure.dpi": 100,                  # Display DPI (use 300 for saving)

    "font.size": 16,                    # Base font size
    "axes.titlesize": 22,               # Title font size
    "axes.labelsize": 18,               # Axis label font size
    "xtick.labelsize": 14,              # X-axis tick font size
    "ytick.labelsize": 14,              # Y-axis tick font size
    "legend.fontsize": 15,              # Legend text font size
    "legend.title_fontsize": 16,        # Legend title font size

    "lines.linewidth": 3,
    "lines.markersize": 8,

    "legend.frameon": False,            # No box around legend
    "legend.loc": "best",               # Auto position

    "axes.grid": True,
    "grid.alpha": 0.4,
    "grid.linestyle": "--",

    "savefig.dpi": 300,                 # DPI when saving
    "savefig.bbox": "tight",           # Trim edges
})

fig = plt.figure()
plt.plot(train_counter, train_losses, color='blue')
plt.scatter(test_counter, test_losses, color='red')
plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
plt.xlabel('Number of training examples seen')
plt.ylabel('Negative log likelihood loss')
plt.title('Training and Test Loss')
plt.savefig(os.path.join(figures_folder, 'loss_plot.png'))
plt.close(fig)