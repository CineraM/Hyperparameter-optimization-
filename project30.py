'''
linux stuff:
nvidia-smi
conda env list
conda install "asdsa"
conda activate nameofenv
'''
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
import torchvision
import torchvision.transforms as transforms
from torch import nn
from torch.nn import functional as F
import matplotlib.pyplot as plt
import time
from IPython import display

net = nn.Sequential(# First convolution layer
                    nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2),
                    nn.Sigmoid(), #activation layer
                    nn.AvgPool2d(kernel_size=2, stride=2),
                    # Second convolution layer
                    nn.Conv2d(6, 16, kernel_size=5),
                    nn.Sigmoid(), #activation layer
                    nn.AvgPool2d(kernel_size=2, stride=2),
                    # Third fully connected (FC) layer
                    nn.Flatten(),
                    nn.Linear(16 * 5 * 5, 120),
                    nn.Sigmoid(),
                    # Fourth fully connected (FC) layer
                    nn.Linear(120, 84),
                    nn.Sigmoid(),
                    # Fifth fully connected (FC) layer
                    nn.Linear(84, 10))
FashionMNIST_classes = ('T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal','Shirt', 'Sneaker', 'Bag', 'Ankle boot')
transform_FashionMNIST = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5), (0.5))])
batch_size = 16
trainset = torchvision.datasets.FashionMNIST(root='./data', train=True,download=True, transform=transform_FashionMNIST)
testset = torchvision.datasets.FashionMNIST(root='./data', train=False,download=True, transform=transform_FashionMNIST)
train_iter = torch.utils.data.DataLoader(trainset, batch_size=batch_size,shuffle=True, num_workers=2)
test_iter = torch.utils.data.DataLoader(testset, batch_size=batch_size,shuffle=False, num_workers=2)
images, labels = next(iter(train_iter)) # get some random training images

def default_parameters():
    net = nn.Sequential(# First convolution layer
                        nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2),
                        nn.Sigmoid(), #activation layer
                        nn.AvgPool2d(kernel_size=2, stride=2),
                        # Second convolution layer
                        nn.Conv2d(6, 16, kernel_size=5),
                        nn.Sigmoid(), #activation layer
                        nn.AvgPool2d(kernel_size=2, stride=2),
                        # Third fully connected (FC) layer
                        nn.Flatten(),
                        nn.Linear(16 * 5 * 5, 120),
                        nn.Sigmoid(),
                        # Fourth fully connected (FC) layer
                        nn.Linear(120, 84),
                        nn.Sigmoid(),
                        # Fifth fully connected (FC) layer
                        nn.Linear(84, 10))
    FashionMNIST_classes = ('T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal',
                            'Shirt', 'Sneaker', 'Bag', 'Ankle boot')
    transform_FashionMNIST = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5), (0.5))])
    batch_size = 16
    trainset = torchvision.datasets.FashionMNIST(root='./data', train=True,
                                            download=True, transform=transform_FashionMNIST)
    testset = torchvision.datasets.FashionMNIST(root='./data', train=False,
                                           download=True, transform=transform_FashionMNIST)

    train_iter = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)
    test_iter = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=2)
    images, labels = next(iter(train_iter)) # get some random training images

class Accumulator:
    """For accumulating sums over `n` variables."""
    def __init__(self, n):
        self.data = [0.0] * n # creates [0.0, 0.0, ....n-times]

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data) # creates [0.0, 0.0, ....n-times]

    def __getitem__(self, idx):
        return self.data[idx]


def accuracy (y_hat, y):
    """Compute the number of correct predictions."""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1: #The output is a vector -- not just a scalar
        y_hat = y_hat.argmax(axis=1)
    cmp = (y_hat.type(y.dtype) == y)
    return float(cmp.type(y.dtype).sum())

def evaluate_accuracy (net, data_iter):
    """Compute the accuracy for a model on a dataset."""
    net.eval()  # Set the model to evaluation mode
    metric = Accumulator(2)  # No. of correct predictions, no. of predictions

    with torch.no_grad():
        for X, y in data_iter: # iterate over the minibatches
            X, y = X.to(device), y.to(device)
            metric.add(accuracy(net(X), y), y.numel(), True)
    return metric[0] / metric[1]


def train_one_epoch (net, train_iter, loss, updater):
    """The training loop defined in Chapter 3."""
    # Set the model to training mode
    net.train()
    # Sum of training loss, sum of training accuracy, no. of examples
    metric = Accumulator(3)
    for X, y in train_iter:
            X, y = X.to(device), y.to(device) # Move data to device - GPU or CPU as set
            # Compute gradients and update parameters
            y_hat = net(X)
            l = loss(y_hat, y)

            # Using PyTorch in-built optimizer & loss criterion
            updater.zero_grad()
            l.backward()
            updater.step()
            metric.add(float(l) * len(y), accuracy(y_hat, y), y.numel(), True)
        # Return training loss and training accuracy
    return metric[0] / metric[2], metric[1] / metric[2]

def train_full_sgd (net, train_iter, test_iter, num_epochs, lr, file="default.png"):
    data = []
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_normal_(m.weight)

    net.apply(init_weights)
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD (net.parameters(), lr=lr)

    for epoch in range(num_epochs):
        start_time = time.time()
        #with torch.cuda.device(0):
        train_metrics = train_one_epoch (net, train_iter, loss, optimizer)
        test_acc = evaluate_accuracy(net, test_iter)
        end_time = time.time() - start_time
        print ('Epoch:', epoch + 1, 'time: %4.3f'%(end_time), '(loss: %4.3f, train acc: %4.3f, test acc: %4.3f)'%(train_metrics[0],  train_metrics[1], test_acc))
        data.append([train_metrics[0],  train_metrics[1], test_acc])

    train_loss, train_acc = train_metrics
    return data

def train_full_rmsprop (net, train_iter, test_iter, num_epochs, lr, file="default.png"):
    data = []
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_normal_(m.weight)

    net.apply(init_weights)
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.RMSprop (net.parameters(), lr=lr)

    for epoch in range(num_epochs):
        start_time = time.time()
        #with torch.cuda.device(0):
        train_metrics = train_one_epoch (net, train_iter, loss, optimizer)
        test_acc = evaluate_accuracy(net, test_iter)
        end_time = time.time() - start_time
        print ('Epoch:', epoch + 1, 'time: %4.3f'%(end_time), '(loss: %4.3f, train acc: %4.3f, test acc: %4.3f)'%(train_metrics[0],  train_metrics[1], test_acc))
        data.append([train_metrics[0],  train_metrics[1], test_acc])

    train_loss, train_acc = train_metrics
    return data

def train_full (net, train_iter, test_iter, num_epochs, lr, file="default.png"):
    data = []
    # Initialize
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_normal_(m.weight)

    net.apply(init_weights)
    # Loss Function
    loss = nn.CrossEntropyLoss()
    # Optimizer
    optimizer = torch.optim.Adam (net.parameters(), lr=lr)

    #-----------------------Iterate over epochs--------------------------------------
    flag = True;
    for epoch in range(num_epochs):
        start_time = time.time()
        train_metrics = train_one_epoch (net, train_iter, loss, optimizer)
        test_acc = evaluate_accuracy(net, test_iter)
        end_time = time.time() - start_time
        print ('Epoch:', epoch + 1, 'time: %4.3f'%(end_time), '(loss: %4.3f, train acc: %4.3f, test acc: %4.3f)'%(train_metrics[0],  train_metrics[1], test_acc))
        data.append([train_metrics[0],  train_metrics[1], test_acc])

    train_loss, train_acc = train_metrics
    return data
    #return (train_loss, train_acc, test_acc, data) # not using any of those varaibles

if torch.cuda.is_available():
  device = torch.device("cuda:0")
else:
  device = torch.device("cpu")

# fully connected layers
###############################################################################################################
#-----------------------------8-1---------------------------------------------------------
default_parameters()
net = nn.Sequential(# First convolution layer
                    nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2),
                    nn.Sigmoid(), #activation layer
                    nn.AvgPool2d(kernel_size=2, stride=2),
                    # Second convolution layer
                    nn.Conv2d(6, 16, kernel_size=5),
                    nn.Sigmoid(), #activation layer
                    nn.AvgPool2d(kernel_size=2, stride=2),
                    # Third fully connected (FC) layer
                    nn.Flatten(),
                    nn.Linear(16 * 5 * 5, 120),
                    nn.Sigmoid(),
                    # Fifth fully connected (FC) layer --> output layers
                    nn.Linear(120, 60))
net.to(device)
total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
print('On:', device, 'Number of parameters to estimate/learn:', total_params)

fflay_1 = train_full (net, train_iter, test_iter, 50, 0.0001, ":(");
#-----------------------------8-1---------------------------------------------------------
#-----------------------------8-2---------------------------------------------------------
default_parameters()
net.to(device)
total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
print('On:', device, 'Number of parameters to estimate/learn:', total_params)
fflay_2 = train_full (net, train_iter, test_iter, 50, 0.0001, ":(");
#-----------------------------8-2---------------------------------------------------------

#-----------------------------8-3---------------------------------------------------------
default_parameters()
net = nn.Sequential(# First convolution layer
                    nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2),
                    nn.Sigmoid(), #activation layer
                    nn.AvgPool2d(kernel_size=2, stride=2),

                    # Second convolution layer
                    nn.Conv2d(6, 16, kernel_size=5),
                    nn.Sigmoid(), #activation layer
                    nn.AvgPool2d(kernel_size=2, stride=2),

                    # Third fully connected (FC) layer
                    nn.Flatten(),
                    nn.Linear(16 * 5 * 5, 120),
                    nn.Sigmoid(),

                    # Fourth fully connected (FC) layer
                    nn.Linear(120, 84),
                    nn.Sigmoid(),

                    nn.Linear(84, 40),
                    nn.Sigmoid(),
                    #3 ff layers
                    # Fifth fully connected (FC) layer --> output layers
                    nn.Linear(40, 10))
net.to(device)
total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
print('On:', device, 'Number of parameters to estimate/learn:', total_params)

fflay_3 = train_full (net, train_iter, test_iter, 50, 0.0001, ":(");
#-----------------------------7-3---------------------------------------------------------
fig, ax = plt.subplots()
x_range = range(0, 50)

plt.xlim(0, 50)
plt.ylim(0, 1)
ax.set_yscale('linear')
ax.set_xscale('linear')

ff1_al = [item[0] for item in fflay_1]
ff1_t_acc = [item[1] for item in fflay_1]
ff1_te_acc = [item[2] for item in fflay_1]

ff2_al = [item[0] for item in fflay_2]
ff2_t_acc = [item[1] for item in fflay_2]
ff2_te_acc = [item[2] for item in fflay_2]

ff3_al = [item[0] for item in fflay_3]
ff3_t_acc = [item[1] for item in fflay_3]
ff3_te_acc = [item[2] for item in fflay_3]

ax.plot(x_range, ff1_al, '-', label='1 loss')
ax.plot(x_range, ff1_t_acc, '--', label='1 train acc')
ax.plot(x_range, ff1_te_acc, '-', label='1 test acc')

ax.plot(x_range, ff2_al, '-', label='2 loss')
ax.plot(x_range, ff2_t_acc, '--', label='2 train acc')
ax.plot(x_range, ff2_te_acc, '-', label='2 test acc')

ax.plot(x_range, ff3_al, '-', label='3 loss')
ax.plot(x_range, ff3_t_acc, '--', label='3 train acc')
ax.plot(x_range, ff3_te_acc, '-', label='3 test acc')

ax.set(xlabel='epoch', title='# of Fully-Connected Layers')
ax.grid()
leg = ax.legend();
fig.savefig("8_num_of_ff_layers")
ax.cla()
###############################################################################################################
