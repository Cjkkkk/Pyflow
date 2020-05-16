import flow.function as F
from flow.tensor import Tensor
from flow.module import Conv2d, Linear, Module
from flow.optim import SGD
from flow.data import MNIST, DataLoader
import numpy as np
import pickle
import argparse

class Net(Module):
    """ConvNet -> Max_Pool -> RELU -> ConvNet -> Max_Pool -> RELU -> FC -> RELU -> FC -> SOFTMAX"""
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = Linear(784, 200)
        self.fc2 = Linear(200, 10)

    def forward(self, x):
        x = F.view(x, (-1, 784))
        x = self.fc1(x)
        x = self.fc2(x)
        return F.log_softmax(x)

def train(args, model, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {}'.format(
                epoch, batch_idx * data.shape[0], len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss))

def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        output = model(data)
        test_loss += F.nll_loss(output, target, reduction='sum')  # sum up batch loss
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += (pred == target.view_as(pred)).sum()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyFlow MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.000001, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()

    train_loader = DataLoader(
        MNIST('./data', train=True, download=True),
        batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(
        MNIST('./data', train=False, download=False),
        batch_size=args.test_batch_size, shuffle=True)

    model = Net()
    optimizer = SGD(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        train(args, model, train_loader, optimizer, epoch)
        test(args, model, test_loader)

    if args.save_model:
        with open("mnist_cnn.pt", "wb") as f:
            pickle.dump(model.state_dict(), f)


if __name__ == '__main__':
    main()