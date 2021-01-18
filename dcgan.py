from __future__ import print_function
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.utils import save_image, make_grid

from einops import rearrange, reduce
from PIL import Image
from sklearn.metrics import confusion_matrix
import boto3
import re
from get_config import get_config
from image_preprocessing import get_cleaned_data
from sklearn.model_selection import train_test_split

base_config = get_config()
config = base_config["neural_net"]["dcgan"]

# add model arguments from config file
parser = argparse.ArgumentParser(description='DCGAN Plays')

# This model is trained via backpropigation in batches. This allows the model to be trained much quicker
# due to the smaller number of observations making the necessary operations less computationally expensive.
# There is also a well established theory that states that since Stochastic Gradient Descent, introduces a small
# generalization error (by introducing randomness) and is a uniformly stable alogrithm, this causes generalization
# in expectation of the test data.
parser.add_argument('--batch-size', type=int, default=config["batch_size"], metavar='N',
                    help='input batch size for training (default: ' + str(config["batch_size"]) +')')

# number of training epochs (trips through entire dataset)
parser.add_argument('--epochs', type=int, default=config["epochs"], metavar='N',
                    help='number of epochs to train (default: ' + str(config["epochs"]) +')')

# whether or not we want to train the model on GPUs if available (Of course we do!)
parser.add_argument('--no-cuda', action='store_true', default=config["no_cuda"],
                    help='enables CUDA training')

# set random seed for reliable model reproduction
parser.add_argument('--seed', type=int, default=config["seed"], metavar='S',
                    help='random seed (default: ' + str(config["seed"]) +')')

# The interval at which our model logs its progress
parser.add_argument('--log-interval', type=int, default=config["log_interval"], metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

# set seed
torch.manual_seed(args.seed)

# use GPU if available, else use CPU
device = torch.device("cuda" if args.cuda else "cpu")
# kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
kwargs = {}

# import our data and split it into training and testing sets
print("importing data")
data = get_cleaned_data(output_path=base_config["data_prep"]["sample_image_path"])
training_data, test_data = train_test_split(data, test_size=config["test_size"]) # this needs to be consistent across models

# convert data to pytorch tensor type
print("cleaning data")
training_data = torch.Tensor(training_data)
test_data = torch.Tensor(test_data)

train_loader = torch.utils.data.DataLoader(training_data, batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=True, **kwargs)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.fc1 = nn.Linear(300, 1200)
        self.fc2 = nn.Linear(1200, 5000)
        self.fc3 = nn.Linear(5000, 240*19*42)

        # (240 x 19 x 42)
        # (240 x 38 x 84)
        self.conv1 = nn.ConvTranspose2d(in_channels=240, out_channels=120, kernel_size=(2,3), stride=1, output_padding=0)

        # (120 x 39 x 84)
        # (120 x 78 x 172)
        self.conv2 = nn.ConvTranspose2d(in_channels=120, out_channels=3, kernel_size=(5,7), stride=2, output_padding=1)

    def forward(self, z):
        # add two fully connected layers
        lin1 = F.relu(self.fc1(z))
        lin2 = F.relu(self.fc2(lin1))
        lin3 = F.relu(self.fc3(lin2))
        lin3 = lin3.reshape(-1, 240, 19, 42)
        max1 = F.interpolate(lin3, scale_factor=2)
        con3 = F.relu(self.conv1(max1))
        max2 = F.interpolate(con3, scale_factor=2)
        return torch.sigmoid(self.conv2(max2))

    
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        # encode image
        # input image dimesions (3 x 160 x 350)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=120, kernel_size=(5,7), stride=2)

        # convolution height (160-5+1)/2 = 78, convolution width (350-7+1)/2 = 172 (120 x 78 x 172)
        # max pooling height floor(78/2) = 39 , max pooling width floor(172/2) = 86 (120 x 39 x 86)
        self.conv2 = nn.Conv2d(in_channels=120, out_channels=240, kernel_size=(2,3), stride=1)

        self.fc1 = nn.Linear(240*19*42, 5000)
        self.fc2 = nn.Linear(5000, 1200)
        self.fc3 = nn.Linear(1200, 300)
        self.fc4 = nn.Linear(300, 1)

    def forward(self, x):
        con1 = F.relu(self.conv1(x))
        max1 = F.max_pool2d(con1, 2, 2)
        con2 = F.relu(self.conv2(max1))
        max2 = F.max_pool2d(con2, 2, 2)
        lin0 = max2.view(-1,19*42*240)
        lin1 = F.relu(self.fc1(lin0))
        lin2 = F.relu(self.fc2(lin1))
        lin3 = F.relu(self.fc3(lin2))
        return torch.sigmoid(self.fc4(lin3))


print("save models to device and define optimizer as ADAM")
generator = Generator().to(device)
G_optimizer = optim.Adam(generator.parameters(), lr=1e-3)

discriminator = Discriminator().to(device)
D_optimizer = optim.Adam(discriminator.parameters(), lr=1e-3)
print("done")

# binary cross entropy: -(p(x)*log(q(x)) + (1-p(x))*log(1-q(x)))
criterion = nn.BCELoss()

real_label = 1.
fake_label = 0.

def train():
    """
    This function trains each of the models.
    
    The first thing that is done is training the Discriminator to correctly images as real or fake. This means
    maximizing log(D(x)) + log(1-D(G(z))). We will calculate this in 2 steps due to the suggestion from ganhacks.
    First, we will evaluate a real batch with the loss -log(D(x)), then calculate the gradients.
    Next, we will evaluate a batch of 'fake' images generated by the generator with the loss -log(1-D(G(z))). These gradients
    are then calculated with a backward pass from the all-real and all-fake samples. The ADAM optimization algorithm is used
    to change the weights of the dsicriminator netowrk

    The next thing that is done is training the Generator. The Generator will look to minimize log(1-D(G(z))), which
    directly contradicts what we want to minimize in the Discriminator. This is accomplished by classifying the generator output in
    part 1 of the discriminator training but using real labels
    """

    # switch model to training mode
    discriminator.train()
    generator.train()

    # initialize training loss
    D_loss = 0
    G_Loss = 0

    for epoch in range(1, args.epochs + 1):
        print(str(epoch))
        for batch_idx, data in enumerate(train_loader):
            #########################################################################
            # (1) Update Discriminator network: maximize log(D(x)) + log(1 - D(G(z)))
            #########################################################################

            # Train with all real samples
            # save batch to device
            data = data.to(device)

            # zero out the gradient for each iteration
            D_optimizer.zero_grad()

            # get labels for real data and run discriminator on real samples
            label = torch.full((len(data),), real_label, dtype=torch.float)
            output = discriminator(data).view(-1)

            # Calculate loss on all-real batch and get the gradients for the discriminator
            print(output.shape)
            print(label.shape)
            errD_real = criterion(output, label)
            errD_real.backward()

            # Use gaussian noise to feed to Generator to generate all fakes sample
            gaussian_noise = torch.randn(len(data), 300)
            fake_data = generator(gaussian_noise)
            print(gaussian_noise.shape)
            print(fake_data.shape)

            # create labels for fake data
            label = torch.full((len(data),), fake_label, dtype=torch.float)

            # classify the fake samples
            output = discriminator(fake_data.detach()).view(-1)

            # Calculate loss on all-fake batch and get the gradients for the discriminator
            errD_fake = criterion(output, label)
            errD_fake.backward()

            # add to total discriminator error
            D_loss = D_loss + errD_real + errD_fake

            # update the Discriminator Network weights using ADAM method 
            D_optimizer.step()

            #############################################
            # (2) Update G network: maximize log(D(G(z)))
            #############################################
            # zero out generator gradient
            G_optimizer.zero_grad()

            # label the fake data as real to optimize the generator into tricking the Discriminator
            # in regards to which data is real
            label = torch.full((len(data),), real_label, dtype=torch.float)

            # discriminator predicts on fake data
            output = discriminator(fake_data).view(-1)

            # get BCE for generator and associated gradients
            errG = criterion(output, label)
            errG.backward()
            
            # add to total discriminator error
            G_loss = G_loss + errG

            # update the Discriminator Network weights using ADAM method 
            G_optimizer.step()

        if epoch % args.log_interval == 0:
            print('====> Epoch: {} Average Generator loss: {:.4f}'.format(epoch, G_loss / len(train_loader.dataset)))
            print('====> Epoch: {} Average Discriminator loss: {:.4f}'.format(epoch, D_loss / len(train_loader.dataset)))

if __name__ == '__main__':
    print("Start")
    train()