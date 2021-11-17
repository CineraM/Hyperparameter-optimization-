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

# optimization functions
###############################################################################################################
#-----------------------------1-SGD---------------------------------------------------------
default_parameters()
net.to(device)
total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
print('On:', device, 'Number of parameters to estimate/learn:', total_params)

optim_sgd = train_full_sgd (net, train_iter, test_iter, 50, 0.0001, "remove this later");
#-----------------------------1-SGD---------------------------------------------------------
#-----------------------------1-ADAM---------------------------------------------------------
default_parameters()
net.to(device)
total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
print('On:', device, 'Number of parameters to estimate/learn:', total_params)

optim_adam = train_full (net, train_iter, test_iter, 50, 0.0001, "remove this later");
#-----------------------------1-ADAM---------------------------------------------------------
#-----------------------------1-RMSprop---------------------------------------------------------
default_parameters()
net.to(device)
total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
print('On:', device, 'Number of parameters to estimate/learn:', total_params)

optim_rms = train_full_rmsprop (net, train_iter, test_iter, 50, 0.0001, "remove this later");
#-----------------------------1-RMSprop---------------------------------------------------------
fig, ax = plt.subplots()
x_range = range(0, 50)

plt.xlim(0, 50)
plt.ylim(0, 1)
ax.set_yscale('linear')
ax.set_xscale('linear')

sgd_al = [item[0] for item in optim_sgd]
sgd_t_acc = [item[1] for item in optim_sgd]
sgd_te_acc = [item[2] for item in optim_sgd]

adam_al = [item[0] for item in optim_adam]
adam_t_acc = [item[1] for item in optim_adam]
adam_te_acc = [item[2] for item in optim_adam]

rms_al = [item[0] for item in optim_rms]
rms_t_acc = [item[1] for item in optim_adam]
rms_te_acc = [item[2] for item in optim_adam]

ax.plot(x_range, rms_al, '-', label='RMS loss')
ax.plot(x_range, rms_t_acc, '--', label='RMS train acc')
ax.plot(x_range, rms_te_acc, '-', label='RMS test acc')

ax.plot(x_range, sgd_al, '-', label='SGD loss')
ax.plot(x_range, sgd_t_acc, '--', label='SGD train acc')
ax.plot(x_range, sgd_te_acc, '-', label='SGD test acc')

ax.plot(x_range, adam_al, '-', label='Adam loss')
ax.plot(x_range, adam_t_acc, '--', label='Adam train acc')
ax.plot(x_range, adam_te_acc, '-', label='Adam test acc')

ax.set(xlabel='epoch', title='Optimization Functions')

ax.grid()
leg = ax.legend();
fig.savefig("1_optimization")
ax.cla()
###############################################################################################################

#BatchSize
###############################################################################################################
#-----------------------------2-16---------------------------------------------------------
default_parameters();
batch_size = 16
net.to(device)
total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
print('On:', device, 'Number of parameters to estimate/learn:', total_params)

b_size16 = train_full (net, train_iter, test_iter, 50, 0.0001, "remove this later");
#-----------------------------2-16---------------------------------------------------------
#-----------------------------2-32---------------------------------------------------------
default_parameters();
batch_size = 32
net.to(device)
total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
print('On:', device, 'Number of parameters to estimate/learn:', total_params)

b_size32 = train_full (net, train_iter, test_iter, 50, 0.0001, "remove this later");
#-----------------------------2-32---------------------------------------------------------
#-----------------------------2-64---------------------------------------------------------
default_parameters();
batch_size = 64
net.to(device)
total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
print('On:', device, 'Number of parameters to estimate/learn:', total_params)

b_size64 = train_full (net, train_iter, test_iter, 50, 0.0001, ":(");
#-----------------------------2-64---------------------------------------------------------
#-----------------------------2-256---------------------------------------------------------
default_parameters();
batch_size = 256
net.to(device)
total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
print('On:', device, 'Number of parameters to estimate/learn:', total_params)

b_size256 = train_full (net, train_iter, test_iter, 50, 0.0001, ":(");
#-----------------------------2-64---------------------------------------------------------

fig, ax = plt.subplots()
x_range = range(0, 50)

plt.xlim(0, 50)
plt.ylim(0, 1)
ax.set_yscale('linear')
ax.set_xscale('linear')

bs16_al = [item[0] for item in b_size16]
bs16_t_acc = [item[1] for item in b_size16]
bs16_te_acc = [item[2] for item in b_size16]

bs32_al = [item[0] for item in b_size32]
bs32_t_acc = [item[1] for item in b_size32]
bs32_te_acc = [item[2] for item in b_size32]

bs64_al = [item[0] for item in b_size64]
bs64_t_acc = [item[1] for item in b_size64]
bs64_te_acc = [item[2] for item in b_size64]

bs256_al = [item[0] for item in b_size256]
bs256_t_acc = [item[1] for item in b_size256]
bs256_te_acc = [item[2] for item in b_size256]

ax.plot(x_range, bs16_al, '-', label='BS16 loss')
ax.plot(x_range, bs16_t_acc, '--', label='BS16 train acc')
ax.plot(x_range, bs16_te_acc, '-', label='BS16 test acc')

ax.plot(x_range, bs32_al, '-', label='BS32 loss')
ax.plot(x_range, bs32_t_acc, '--', label='BS32 train acc')
ax.plot(x_range, bs32_te_acc, '-', label='BS32 test acc')

ax.plot(x_range, bs64_al, '-', label='BS64 loss')
ax.plot(x_range, bs64_t_acc, '--', label='BS64 train acc')
ax.plot(x_range, bs64_te_acc, '-', label='BS64 test acc')

ax.plot(x_range, bs256_al, '-', label='BS256 loss')
ax.plot(x_range, bs256_t_acc, '--', label='BS256 train acc')
ax.plot(x_range, bs256_te_acc, '-', label='BS256 test acc')

ax.set(xlabel='epoch', title='Batch Size')
ax.grid()
leg = ax.legend();
fig.savefig("2_batch_size")
ax.cla()
###############################################################################################################


# kernel sizes
###############################################################################################################
#-----------------------------3-3---------------------------------------------------------
default_parameters()
net = nn.Sequential(# First convolution layer
                    nn.Conv2d(in_channels=1, out_channels=6, kernel_size=3, padding=2),
                    nn.Sigmoid(), #activation layer
                    nn.AvgPool2d(kernel_size=2, stride=2),
                    # Second convolution layer
                    nn.Conv2d(6, 16, kernel_size=3),
                    nn.Sigmoid(), #activation layer
                    nn.AvgPool2d(kernel_size=2, stride=2),
                    # Third fully connected (FC) layer
                    nn.Flatten(),
                    nn.Linear(16 * 6 * 6, 120),
                    nn.Sigmoid(),
                    # Fourth fully connected (FC) layer
                    nn.Linear(120, 84),
                    nn.Sigmoid(),
                    # Fifth fully connected (FC) layer
                    nn.Linear(84, 10))
net.to(device)
total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
print('On:', device, 'Number of parameters to estimate/learn:', total_params)

kernel_size_3 = train_full (net, train_iter, test_iter, 50, 0.0001, "remove this later");
#-----------------------------3-3---------------------------------------------------------
#-----------------------------3-5---------------------------------------------------------
default_parameters() # 5 is the default
net.to(device)
total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
print('On:', device, 'Number of parameters to estimate/learn:', total_params)

kernel_size_5 = train_full (net, train_iter, test_iter, 50, 0.0001, "remove this later");
#-----------------------------3-5---------------------------------------------------------
#-----------------------------3-7---------------------------------------------------------
default_parameters()
net = nn.Sequential(# First convolution layer
                    nn.Conv2d(in_channels=1, out_channels=6, kernel_size=7, padding=2),
                    nn.Sigmoid(), #activation layer
                    nn.AvgPool2d(kernel_size=2, stride=2),
                    # Second convolution layer
                    nn.Conv2d(6, 16, kernel_size=7),
                    nn.Sigmoid(), #activation layers
                    nn.AvgPool2d(kernel_size=2, stride=2),
                    # Third fully connected (FC) layer
                    nn.Flatten(),
                    nn.Linear(16 * 3 * 3, 120),
                    nn.Sigmoid(),
                    # Fourth fully connected (FC) layer
                    nn.Linear(120, 84),
                    nn.Sigmoid(),
                    # Fifth fully connected (FC) layer
                    nn.Linear(84, 10))
net.to(device)
total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
print('On:', device, 'Number of parameters to estimate/learn:', total_params)

kernel_size_7 = train_full (net, train_iter, test_iter, 50, 0.0001, "remove this later");
#-----------------------------3-7---------------------------------------------------------
fig, ax = plt.subplots()
x_range = range(0, 50)

plt.xlim(0, 50)
plt.ylim(0, 1)
ax.set_yscale('linear')
ax.set_xscale('linear')

ks3_al = [item[0] for item in kernel_size_3]
ks3_t_acc = [item[1] for item in kernel_size_3]
ks3_te_acc = [item[2] for item in kernel_size_3]

ks5_al = [item[0] for item in kernel_size_5]
ks5_t_acc = [item[1] for item in kernel_size_5]
ks5_te_acc = [item[2] for item in kernel_size_5]

ks7_al = [item[0] for item in kernel_size_7]
ks7_t_acc = [item[1] for item in kernel_size_7]
ks7_te_acc = [item[2] for item in kernel_size_7]

ax.plot(x_range, ks3_al, '-', label='K3 loss')
ax.plot(x_range, ks3_t_acc, '--', label='K3 train acc')
ax.plot(x_range, ks3_te_acc, '-', label='K3 test acc')

ax.plot(x_range, ks5_al, '-', label='K5 loss')
ax.plot(x_range, ks5_t_acc, '--', label='K5 train acc')
ax.plot(x_range, ks5_te_acc, '-', label='K5 test acc')

ax.plot(x_range, ks7_al, '-', label='K7 loss')
ax.plot(x_range, ks7_t_acc, '--', label='K7 train acc')
ax.plot(x_range, ks7_te_acc, '-', label='K7 test acc')

ax.set(xlabel='epoch', title='Convolution kernel sizes')

ax.grid()
leg = ax.legend();
fig.savefig("3_kernel_sizes")
ax.cla()
###############################################################################################################


#Output Channels
###############################################################################################################
#-----------------------------4-(4, 12)---------------------------------------------------------
default_parameters()
net = nn.Sequential(# First convolution layer
                    nn.Conv2d(in_channels=1, out_channels=4, kernel_size=5, padding=2),
                    nn.Sigmoid(), #activation layer
                    nn.AvgPool2d(kernel_size=2, stride=2),
                    # Second convolution layer
                    nn.Conv2d(4, 12, kernel_size=5),
                    nn.Sigmoid(), #activation layer
                    nn.AvgPool2d(kernel_size=2, stride=2),
                    # Third fully connected (FC) layer
                    nn.Flatten(),
                    nn.Linear(12 * 5 * 5, 120),
                    nn.Sigmoid(),
                    # Fourth fully connected (FC) layer
                    nn.Linear(120, 84),
                    nn.Sigmoid(),
                    # Fifth fully connected (FC) layer
                    nn.Linear(84, 10))
net.to(device)
total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
print('On:', device, 'Number of parameters to estimate/learn:', total_params)

outc_4_12 = train_full (net, train_iter, test_iter, 50, 0.0001, ":(");
#-----------------------------4-(4, 12)---------------------------------------------------------
#-----------------------------4-(6, 16)---------------------------------------------------------
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
                    # Fifth fully connected (FC) layer
                    nn.Linear(84, 10))
net.to(device)
total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
print('On:', device, 'Number of parameters to estimate/learn:', total_params)

outc_6_16 = train_full (net, train_iter, test_iter, 50, 0.0001, ":(");
#-----------------------------4-(6, 16)---------------------------------------------------------
#-----------------------------4-(8, 20)---------------------------------------------------------
default_parameters()
net = nn.Sequential(# First convolution layer
                    nn.Conv2d(in_channels=1, out_channels=8, kernel_size=5, padding=2),
                    nn.Sigmoid(), #activation layer
                    nn.AvgPool2d(kernel_size=2, stride=2),
                    # Second convolution layer
                    nn.Conv2d(8, 20, kernel_size=5),
                    nn.Sigmoid(), #activation layer
                    nn.AvgPool2d(kernel_size=2, stride=2),
                    # Third fully connected (FC) layer
                    nn.Flatten(),
                    nn.Linear(20 * 5 * 5, 120),
                    nn.Sigmoid(),
                    # Fourth fully connected (FC) layer
                    nn.Linear(120, 84),
                    nn.Sigmoid(),
                    # Fifth fully connected (FC) layer
                    nn.Linear(84, 10))
net.to(device)
total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
print('On:', device, 'Number of parameters to estimate/learn:', total_params)

outc_8_20 = train_full (net, train_iter, test_iter, 50, 0.0001, ":(");
#-----------------------------4-(6, 16)---------------------------------------------------------
fig, ax = plt.subplots()
x_range = range(0, 50)

plt.xlim(0, 50)
plt.ylim(0, 1)
ax.set_yscale('linear')
ax.set_xscale('linear')

oc_4_12_al = [item[0] for item in outc_4_12]
oc_4_12_t_acc = [item[1] for item in outc_4_12]
oc_4_12_te_acc = [item[2] for item in outc_4_12]

oc_6_16_al = [item[0] for item in outc_6_16]
oc_6_16_t_acc = [item[1] for item in outc_6_16]
oc_6_16_te_acc = [item[2] for item in outc_6_16]

oc_8_20_al = [item[0] for item in outc_8_20]
oc_8_20_t_acc = [item[1] for item in outc_8_20]
oc_8_20_te_acc = [item[2] for item in outc_8_20]

ax.plot(x_range, oc_4_12_al, '-', label='(4, 12) loss')
ax.plot(x_range, oc_4_12_t_acc, '--', label='(4, 12) train acc')
ax.plot(x_range, oc_4_12_te_acc, '-', label='(4, 12) test acc')

ax.plot(x_range, oc_6_16_al, '-', label='(6, 16) loss')
ax.plot(x_range, oc_6_16_t_acc, '--', label='(6, 16) train acc')
ax.plot(x_range, oc_6_16_te_acc, '-', label='(6, 16) test acc')

ax.plot(x_range, oc_8_20_al, '-', label='(8, 20) loss')
ax.plot(x_range, oc_8_20_t_acc, '--', label='(8, 20) train acc')
ax.plot(x_range, oc_8_20_te_acc, '-', label='(8, 20) test acc')

ax.set(xlabel='epoch', title='Number of output channels')
ax.grid()
leg = ax.legend();
fig.savefig("4_output_channels")
ax.cla()
###############################################################################################################

# pooling
###############################################################################################################
#-----------------------------5-avg---------------------------------------------------------
default_parameters() #avg pooling is the default
net.to(device)
total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
print('On:', device, 'Number of parameters to estimate/learn:', total_params)

pooling_avg = train_full (net, train_iter, test_iter, 50, 0.0001, "remove this later");
#-----------------------------5-avg---------------------------------------------------------
#-----------------------------5-max---------------------------------------------------------
default_parameters()
net = nn.Sequential(# First convolution layer
                    nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2),
                    nn.Sigmoid(), #activation layer
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    # Second convolution layer
                    nn.Conv2d(6, 16, kernel_size=5),
                    nn.Sigmoid(), #activation layer
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    # Third fully connected (FC) layer
                    nn.Flatten(),
                    nn.Linear(16 * 5 * 5, 120),
                    nn.Sigmoid(),
                    # Fourth fully connected (FC) layer
                    nn.Linear(120, 84),
                    nn.Sigmoid(),
                    # Fifth fully connected (FC) layer
                    nn.Linear(84, 10))
net.to(device)
total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
print('On:', device, 'Number of parameters to estimate/learn:', total_params)

pooling_max = train_full (net, train_iter, test_iter, 50, 0.0001, "remove this later");
#-----------------------------5-max---------------------------------------------------------

fig, ax = plt.subplots()
x_range = range(0, 50)

plt.xlim(0, 50)
plt.ylim(0, 1)
ax.set_yscale('linear')
ax.set_xscale('linear')

p_avg_al = [item[0] for item in pooling_avg]
p_avg_t_acc = [item[1] for item in pooling_avg]
p_avg_te_acc = [item[2] for item in pooling_avg]

p_max_al = [item[0] for item in pooling_max]
p_max_t_acc = [item[1] for item in pooling_max]
p_max_te_acc = [item[2] for item in pooling_max]

ax.plot(x_range, p_avg_al, '-', label='avg loss')
ax.plot(x_range, p_avg_t_acc, '--', label='avg train acc')
ax.plot(x_range, p_avg_te_acc, '-', label='avg test acc')

ax.plot(x_range, p_max_al, '-', label='max loss')
ax.plot(x_range, p_max_t_acc, '--', label='max train acc')
ax.plot(x_range, p_max_te_acc, '-', label='max test acc')

ax.set(xlabel='epoch', title='Pooling')
ax.grid()
leg = ax.legend();
fig.savefig("5_pooling")
ax.cla()
###############################################################################################################


# activation functions
###############################################################################################################
#-----------------------------6-softmax---------------------------------------------------------
default_parameters()
net = nn.Sequential(# First convolution layer
                    nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2),
                    nn.Softmax(), #activation layer
                    nn.AvgPool2d(kernel_size=2, stride=2),
                    # Second convolution layer
                    nn.Conv2d(6, 16, kernel_size=5),
                    nn.Softmax(), #activation layer
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
net.to(device)
total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
print('On:', device, 'Number of parameters to estimate/learn:', total_params)

afnc_softmax = train_full (net, train_iter, test_iter, 50, 0.0001, "remove this later");
#-----------------------------6-softmax---------------------------------------------------------
#-----------------------------6-ReLU---------------------------------------------------------
default_parameters()
net = nn.Sequential(# First convolution layer
                    nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2),
                    nn.ReLU(), #activation layer
                    nn.AvgPool2d(kernel_size=2, stride=2),
                    # Second convolution layer
                    nn.Conv2d(6, 16, kernel_size=5),
                    nn.ReLU(), #activation layer
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
net.to(device)
total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
print('On:', device, 'Number of parameters to estimate/learn:', total_params)

afnc_relu = train_full (net, train_iter, test_iter, 50, 0.0001, "remove this later");
#-----------------------------6-ReLU---------------------------------------------------------
#-----------------------------6-LeakyReLU---------------------------------------------------------
default_parameters()
net = nn.Sequential(# First convolution layer
                    nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2),
                    nn.LeakyReLU(), #activation layer
                    nn.AvgPool2d(kernel_size=2, stride=2),
                    # Second convolution layer
                    nn.Conv2d(6, 16, kernel_size=5),
                    nn.LeakyReLU(), #activation layer
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
net.to(device)
total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
print('On:', device, 'Number of parameters to estimate/learn:', total_params)

afnc_LeakyReLU = train_full (net, train_iter, test_iter, 50, 0.0001, "remove this later");
#-----------------------------6-LeakyReLU---------------------------------------------------------
fig, ax = plt.subplots()
x_range = range(0, 50)

plt.xlim(0, 50)
plt.ylim(0, 1)
ax.set_yscale('linear')
ax.set_xscale('linear')

ac_smax_al = [item[0] for item in afnc_softmax]
ac_smax_t_acc = [item[1] for item in afnc_softmax]
ac_smax_te_acc = [item[2] for item in afnc_softmax]

ac_relu_al = [item[0] for item in afnc_relu]
ac_relu_t_acc = [item[1] for item in afnc_relu]
ac_relu_te_acc = [item[2] for item in afnc_relu]

ac_leaky_al = [item[0] for item in afnc_LeakyReLU]
ac_leaky_t_acc = [item[1] for item in afnc_LeakyReLU]
ac_leaky_te_acc = [item[2] for item in afnc_LeakyReLU]

ax.plot(x_range, ac_smax_al, '-', label='Softmax loss')
ax.plot(x_range, ac_smax_t_acc, '--', label='softmax train acc')
ax.plot(x_range, ac_smax_te_acc, '-', label='softmax test acc')

ax.plot(x_range, ac_relu_al, '-', label='Relu loss')
ax.plot(x_range, ac_relu_t_acc, '--', label='Relu train acc')
ax.plot(x_range, ac_relu_te_acc, '-', label='Relu test acc')

ax.plot(x_range, ac_leaky_al, '-', label='Lrelu loss')
ax.plot(x_range, ac_leaky_t_acc, '--', label='Lrelu train acc')
ax.plot(x_range, ac_leaky_te_acc, '-', label='Lrelu test acc')

ax.set(xlabel='epoch', title='Activation Functions')

ax.grid()
leg = ax.legend();
fig.savefig("6_activation_functions")
ax.cla()
###############################################################################################################

# # of conv layers
###############################################################################################################
#-----------------------------7-3---------------------------------------------------------
default_parameters()
net = nn.Sequential(# First convolution layer
                    nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2),
                    nn.Sigmoid(), #activation layer
                    nn.AvgPool2d(kernel_size=2, stride=2),
                    # Second convolution layer
                    nn.Conv2d(6, 16, kernel_size=5),
                    nn.Sigmoid(), #activation layer
                    nn.AvgPool2d(kernel_size=2, stride=2),

                    nn.Conv2d(16, 32, kernel_size=5),
                    nn.Sigmoid(), #activation layer
                    # Third fully connected (FC) layer
                    nn.Flatten(),
                    nn.Linear(32, 120),
                    nn.Sigmoid(),
                    # Fourth fully connected (FC) layer
                    nn.Linear(120, 84),
                    nn.Sigmoid(),
                    # Fifth fully connected (FC) layer
                    nn.Linear(84, 10))
net.to(device)
total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
print('On:', device, 'Number of parameters to estimate/learn:', total_params)

clayers_3 = train_full (net, train_iter, test_iter, 50, 0.0001, ":(");
#-----------------------------7-3---------------------------------------------------------
#-----------------------------7-2---------------------------------------------------------
default_parameters()
net.to(device)
total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
print('On:', device, 'Number of parameters to estimate/learn:', total_params)
clayers_2 = train_full (net, train_iter, test_iter, 50, 0.0001, ":(");
#-----------------------------7-2---------------------------------------------------------
#-----------------------------7-1---------------------------------------------------------
default_parameters()
net = nn.Sequential(# First convolution layer
                    nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2),
                    nn.Sigmoid(), #activation layer
                    nn.AvgPool2d(kernel_size=2, stride=2),

                    nn.Flatten(),
                    nn.Linear(1176, 120),
                    nn.Sigmoid(),

                    nn.Linear(120, 84),
                    nn.Sigmoid(),

                    nn.Linear(84, 10))
net.to(device)
total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
print('On:', device, 'Number of parameters to estimate/learn:', total_params)

clayers_1 = train_full (net, train_iter, test_iter, 50, 0.0001, ":(");
#-----------------------------7-1---------------------------------------------------------
fig, ax = plt.subplots()
x_range = range(0, 50)

plt.xlim(0, 50)
plt.ylim(0, 1)
ax.set_yscale('linear')
ax.set_xscale('linear')

clay_3_al = [item[0] for item in clayers_3]
clay_3_t_acc = [item[1] for item in clayers_3]
clay_3_te_acc = [item[2] for item in clayers_3]

clay_2_al = [item[0] for item in clayers_2]
clay_2_al_t_acc = [item[1] for item in clayers_2]
clay_2_al_te_acc = [item[2] for item in clayers_2]

clay_1_al = [item[0] for item in clayers_1]
clay_1_t_acc = [item[1] for item in clayers_1]
clay_1_te_acc = [item[2] for item in clayers_1]

ax.plot(x_range, clay_1_al, '-', label='1 loss')
ax.plot(x_range, clay_1_t_acc, '--', label='1 train acc')
ax.plot(x_range, clay_1_te_acc, '-', label='1 test acc')

ax.plot(x_range, clay_2_al, '-', label='2 loss')
ax.plot(x_range, clay_2_al_t_acc, '--', label='2 train acc')
ax.plot(x_range, clay_2_al_te_acc, '-', label='2 test acc')

ax.plot(x_range, clay_3_al, '-', label='3 loss')
ax.plot(x_range, clay_3_t_acc, '--', label='3 train acc')
ax.plot(x_range, clay_3_te_acc, '-', label='3 test acc')

ax.set(xlabel='epoch', title='# of Convolution Layers')
ax.grid()
leg = ax.legend();
fig.savefig("7_num_of_conv_layers")
ax.cla()
###############################################################################################################


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
                    nn.Sigmoid())
net.to(device)
total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
print('On:', device, 'Number of parameters to estimate/learn:', total_params)

fflay_1 = train_full (net, train_iter, test_iter, 50, 0.0001, ":(");
#-----------------------------8-1---------------------------------------------------------
#-----------------------------8-2---------------------------------------------------------
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
                    nn.Sigmoid())
net.to(device)
total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
print('On:', device, 'Number of parameters to estimate/learn:', total_params)
fflay_2 = train_full (net, train_iter, test_iter, 50, 0.0001, ":(");
#-----------------------------8-2---------------------------------------------------------
#-----------------------------8-3---------------------------------------------------------
default_parameters()
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

# learning rates and epochs
###############################################################################################################
#-----------------------------9-0.00001, 1000---------------------------------------------------------
default_parameters()
net.to(device)
total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
print('On:', device, 'Number of parameters to estimate/learn:', total_params)

lr_e_1000 = train_full (net, train_iter, test_iter, 1000, 0.00001, "remove this later");
#-----------------------------9-0.00001, 1000---------------------------------------------------------
#-----------------------------9-0.0001, 500---------------------------------------------------------
default_parameters()
net.to(device)
total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
print('On:', device, 'Number of parameters to estimate/learn:', total_params)

lr_e_500 = train_full (net, train_iter, test_iter, 500, 0.0001, "remove this later");
#-----------------------------9-0.0001, 500---------------------------------------------------------
#-----------------------------9-0.001, 250---------------------------------------------------------
default_parameters()
net.to(device)
total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
print('On:', device, 'Number of parameters to estimate/learn:', total_params)

lr_e_250 = train_full (net, train_iter, test_iter, 250, 0.001, "remove this later");
#-----------------------------9-0.001, 250---------------------------------------------------------
#-----------------------------9-0.01, 100---------------------------------------------------------
default_parameters()
net.to(device)
total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
print('On:', device, 'Number of parameters to estimate/learn:', total_params)

lr_e_100 = train_full (net, train_iter, test_iter, 100, 0.01, "remove this later");
#-----------------------------9-0.01, 100---------------------------------------------------------
#-----------------------------9-0.1, 50---------------------------------------------------------
default_parameters()
net.to(device)
total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
print('On:', device, 'Number of parameters to estimate/learn:', total_params)

lr_e_50 = train_full (net, train_iter, test_iter, 50, 0.1, "remove this later");
#-----------------------------9-0.1, 50---------------------------------------------------------

fig, ax = plt.subplots()
x_range = range(0, 50)

plt.xlim(0, 1000)
plt.ylim(0, 1)
ax.set_yscale('linear')
ax.set_xscale('linear')

le_1000_al = [item[0] for item in lr_e_1000]
le_1000_t_acc = [item[1] for item in lr_e_1000]
le_1000_te_acc = [item[2] for item in lr_e_1000]

le_500_al = [item[0] for item in lr_e_500]
le_500_acc = [item[1] for item in lr_e_500]
le_500_te_acc = [item[2] for item in lr_e_500]

le_250_al = [item[0] for item in lr_e_250]
le_250_t_acc = [item[1] for item in lr_e_250]
le_250_te_acc = [item[2] for item in lr_e_250]

le_100_al = [item[0] for item in lr_e_100]
le_100_t_acc = [item[1] for item in lr_e_100]
le_100_te_acc = [item[2] for item in lr_e_100]

le_50_al = [item[0] for item in lr_e_50]
le_50_t_acc = [item[1] for item in lr_e_50]
le_50_te_acc = [item[2] for item in lr_e_50]

ax.plot(range(0, 1000), le_1000_al, '-', label='1000 loss')
ax.plot(range(0, 1000), le_1000_t_acc, '--', label='1000 train acc')
ax.plot(range(0, 1000), le_1000_te_acc, '-', label='1000 test acc')

ax.plot(range(0, 500), le_500_al, '-', label='500 loss')
ax.plot(range(0, 500), le_500_acc, '--', label='500 train acc')
ax.plot(range(0, 500), le_500_te_acc, '-', label='500 test acc')

ax.plot(range(0, 250), le_250_al, '-', label='250 loss')
ax.plot(range(0, 250), le_250_t_acc, '--', label='250 train acc')
ax.plot(range(0, 250), le_250_te_acc, '-', label='250 test acc')

ax.plot(range(0, 100), le_100_al, '-', label='100 loss')
ax.plot(range(0, 100), le_100_t_acc, '--', label='100 train acc')
ax.plot(range(0, 100), le_100_te_acc, '-', label='100 test acc')

ax.plot(range(0, 50), le_50_al, '-', label='50 loss')
ax.plot(range(0, 50), le_50_t_acc, '--', label='50 train acc')
ax.plot(range(0, 50), le_50_te_acc, '-', label='50 test acc')

ax.set(xlabel='epoch', title='Learning rate & #epochs')
ax.grid()
leg = ax.legend();
fig.savefig("9_learning_rate_&_epochs")
ax.cla()
###############################################################################################################

def train_full_init_default (net, train_iter, test_iter, num_epochs, lr, file="default.png"):
    data = []
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            pass
    net.apply(init_weights)
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam (net.parameters(), lr=lr)
    #-----------------------Iterate over epochs--------------------------------------
    for epoch in range(num_epochs):
        start_time = time.time()
        train_metrics = train_one_epoch (net, train_iter, loss, optimizer)
        test_acc = evaluate_accuracy(net, test_iter)
        end_time = time.time() - start_time
        print ('Epoch:', epoch + 1, 'time: %4.3f'%(end_time), '(loss: %4.3f, train acc: %4.3f, test acc: %4.3f)'%(train_metrics[0],  train_metrics[1], test_acc))
        data.append([train_metrics[0],  train_metrics[1], test_acc])

    train_loss, train_acc = train_metrics
    return data

def train_full_init_kaiming (net, train_iter, test_iter, num_epochs, lr, file="default.png"):
    data = []
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.kaiming_normal_(m.weight)

    net.apply(init_weights)
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam (net.parameters(), lr=lr)
    #-----------------------Iterate over epochs--------------------------------------
    for epoch in range(num_epochs):
        start_time = time.time()
        train_metrics = train_one_epoch (net, train_iter, loss, optimizer)
        test_acc = evaluate_accuracy(net, test_iter)
        end_time = time.time() - start_time
        print ('Epoch:', epoch + 1, 'time: %4.3f'%(end_time), '(loss: %4.3f, train acc: %4.3f, test acc: %4.3f)'%(train_metrics[0],  train_metrics[1], test_acc))
        data.append([train_metrics[0],  train_metrics[1], test_acc])

    train_loss, train_acc = train_metrics
    return data
#Initializations
###############################################################################################################
#-----------------------------10-default---------------------------------------------------------
default_parameters();
net.to(device)
total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
print('On:', device, 'Number of parameters to estimate/learn:', total_params)

init_default = train_full_init_default (net, train_iter, test_iter, 50, 0.0001, "remove this later");
#-----------------------------10-default---------------------------------------------------------
#-----------------------------10-xavier_normal_---------------------------------------------------------
default_parameters();
net.to(device)
total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
print('On:', device, 'Number of parameters to estimate/learn:', total_params)

init_xavier = train_full (net, train_iter, test_iter, 50, 0.0001, "remove this later");
#-----------------------------10-xavier_normal_---------------------------------------------------------
#-----------------------------10-kaiming_normal_---------------------------------------------------------
default_parameters();
net.to(device)
total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
print('On:', device, 'Number of parameters to estimate/learn:', total_params)

init_kaiming = train_full_init_kaiming (net, train_iter, test_iter, 50, 0.0001, "remove this later");
#-----------------------------10-kaiming_normal_---------------------------------------------------------
fig, ax = plt.subplots()
x_range = range(0, 50)

plt.xlim(0, 50)
plt.ylim(0, 1)
ax.set_yscale('linear')
ax.set_xscale('linear')

in_def_al = [item[0] for item in init_default]
in_def_t_acc = [item[1] for item in init_default]
in_def_te_acc = [item[2] for item in init_default]

in_xav_al = [item[0] for item in init_xavier]
in_xav_t_acc = [item[1] for item in init_xavier]
in_xav_te_acc = [item[2] for item in init_xavier]

in_kai_al = [item[0] for item in init_kaiming]
in_kai_t_acc = [item[1] for item in init_kaiming]
in_kai_te_acc = [item[2] for item in init_kaiming]

ax.plot(x_range, in_def_al, '-', label='default loss')
ax.plot(x_range, in_def_t_acc, '--', label='default train acc')
ax.plot(x_range, in_def_te_acc, '-', label='default test acc')

ax.plot(x_range, in_xav_al, '-', label='xavier loss')
ax.plot(x_range, in_xav_t_acc, '--', label='xavier train acc')
ax.plot(x_range, in_xav_te_acc, '-', label='xavier test acc')

ax.plot(x_range, in_kai_al, '-', label='kaiming loss')
ax.plot(x_range, in_kai_t_acc, '--', label='kaiming train acc')
ax.plot(x_range, in_kai_te_acc, '-', label='kaiming test acc')

ax.set(xlabel='epoch', title='Initializations')
ax.grid()
leg = ax.legend();
fig.savefig("10_batch_size")
ax.cla()
###############################################################################################################
