import os
import sys
import shutil
import unittest
import torch
from torch import nn
import torch.nn.functional as F
import torchvision.models
from torchvision import datasets, transforms

import matplotlib
matplotlib.use("Agg")

import hiddenlayer as hl

# Create output and data directories in project root
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(ROOT_DIR, "test_output")
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
DATA_DIR = os.path.join(ROOT_DIR, "test_data")


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
        return F.log_softmax(x, dim=1)


def train(model, device, train_loader, optimizer, epoch):
    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            model.history.log((epoch, batch_idx),
                loss=loss,
                conv1_weight=model.conv1.weight)

            # At the end of each batch
            with model.canvas:
                model.canvas.draw_plot(model.history["loss"])
                model.canvas.draw_hist(model.history["conv1_weight"])
                # TODO: c.draw_image(model.history["conv1_weight"])

            if batch_idx % 100 == 0:
                model.canvas.save(os.path.join(OUTPUT_DIR, "pytorch_train_{}.png").format(epoch))
            model.history.progress()


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


class TestPytorchWatcher(unittest.TestCase):
    def test_train(self):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST(DATA_DIR, train=True, download=True,
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))
                        ])),
            batch_size=64, shuffle=True)
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST(DATA_DIR, train=False, transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))
                        ])),
            batch_size=1000, shuffle=True)

        model = Net().to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

        # Create History object
        model.history = hl.History()
        model.canvas = hl.Canvas()

        for epoch in range(1, 3):
            train(model, device, train_loader, optimizer, epoch)
            test(model, device, test_loader)

        # Clean up
        shutil.rmtree(OUTPUT_DIR)


if __name__ == "__main__":
    unittest.main()
