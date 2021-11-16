'''
linux stuff:
nvidia-smi
conda env list
conda install "asdsa"

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
#import d2l

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

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0))) #reorder into (row, col, color)
    plt.show()

# get some random training images
images, labels = next(iter(train_iter))


class Animator:
    """For plotting data in animation."""
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(10, 10), plot_str="noname.png"):
        # Incrementally plot multiple lines
        if legend is None:
            legend = []

        self.plot_str = plot_str
        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes,]
        # Use a lambda function to capture arguments
        self.config_axes = lambda: self.set_axes(self.axes[
        0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y, flag, file = "def.png"):
        # Add multiple data points into the figure
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n  #repeat x axis values n times, one for each plot
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        #display.display(self.fig)
        #display.clear_output(wait=True)
        if(flag == False):
            self.fig.savefig(self.plot_str)

    def set_axes(self, axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
      """Set the axes for matplotlib. Defined in :numref:`sec_calculus`"""
      axes.set_xlabel(xlabel), axes.set_ylabel(ylabel)
      axes.set_xscale(xscale), axes.set_yscale(yscale)
      axes.set_xlim(xlim),     axes.set_ylim(ylim)
      if legend:
        axes.legend(legend)
      axes.grid()

class Accumulator:
    """For accumulating sums over `n` variables."""
    def __init__(self, n):
        self.data = [0.0] * n # creates [0.0, 0.0, ....n-times]

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]
        # Example: let self.data = [10, 15] and args be [3, 4]
        # zip(self.data, args) will produce the paired iterable [(10, 3), (15, 4)]
        # the for loop with go over these paired entries adding them up to result in [13, 19]

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

#-------------------------------------------------------------------------------
def train_full (net, train_iter, test_iter, num_epochs, lr, file="default.png"):

    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.1, 1.0],
                       legend=['train loss', 'train acc', 'test acc'], plot_str=file)
    #----------------------Initialize-----------------------------------------
    # There is a default initialization in pytorch uses a uniform distribution bounded by 1/sqrt(in_features),
    # However, this might not be the best one to use. Weight initialization has a great impact on the
    # quality of the final network weights.
    # Good discussions on initialization can be found at
    # https://towardsdatascience.com/how-to-initialize-a-neural-network-27564cfb5ffc
    # https://adityassrana.github.io/blog/theory/2020/08/26/Weight-Init.html
    # Below code allows you to use other well known initialization strategies.
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            #nn.init.xavier_uniform_(m.weight)
            nn.init.xavier_normal_(m.weight)
            #nn.init.kaiming_normal_(m.weight) # -- good for networks with RELU
            #nn.init.kaiming_uniform_(m.weight) # -- good for networks with RELU
    # Uncomment the below if you want to do other types of initializations
    net.apply(init_weights)
    #-----------------------Loss Function--------------------------------------
    loss = nn.CrossEntropyLoss()
    #-----------------------Optimizer--------------------------------------
    # There are different kind of optimizers. The one we studied earlier, stochastic gradient descent (SGD)
    # is but only one type. There are more. See https://pytorch.org/docs/stable/optim.html
    # Adam optimizer is one the most popular one.
    #optimizer = torch.optim.SGD (net.parameters(), lr=lr, momentum=0.9)
    optimizer = torch.optim.Adam (net.parameters(), lr=lr)

    #-----------------------Iterate over epochs--------------------------------------
    flag = True;
    for epoch in range(num_epochs):
        start_time = time.time()
        #with torch.cuda.device(0):
        train_metrics = train_one_epoch (net, train_iter, loss, optimizer)
        test_acc = evaluate_accuracy(net, test_iter)
        end_time = time.time() - start_time
        print ('Epoch:', epoch + 1, 'time: %4.3f'%(end_time), '(loss: %4.3f, train acc: %4.3f, test acc: %4.3f)'%(train_metrics[0],  train_metrics[1], test_acc))
        if(epoch == num_epochs-1):
            flag = False
        animator.add(epoch + 1, train_metrics + (test_acc,), flag)

    train_loss, train_acc = train_metrics
    return (train_loss, train_acc, test_acc)


if torch.cuda.is_available():
  device = torch.device("cuda:0")
else:
  device = torch.device("cpu")

net.to(device)
total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)



'''
For Reference
train_full (net, train_iter, test_iter, num_epochs, lr, "file name of chart");
lr = 0.0001 # Possible choices: 0.00001, 0.0001, 0.001, 0.01, 0.1
num_epochs = 2 # # Possible choices: 500, 250, 50 -- inversely related to the learning rate
'''
print('On:', device, 'Number of parameters to estimate/learn:', total_params)

train_full (net, train_iter, test_iter, 50, 0.0001, "testing1");
train_full (net, train_iter, test_iter, 4, 0.0001, "testing2");
