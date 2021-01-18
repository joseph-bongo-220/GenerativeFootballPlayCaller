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
config = base_config["neural_net"]["cvae"]

# add model arguments from config file
parser = argparse.ArgumentParser(description='CVAE Plays')

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

# DESCRIBing THE CNN ARCHITECTURE 
class ConvVAE(nn.Module):
    def __init__(self):
        super(ConvVAE, self).__init__()
        # encode image
        # input image dimesions (3 x 160 x 350)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=120, kernel_size=(5,7), stride=2)

        # convolution height (160-5+1)/2 = 78, convolution width (350-7+1)/2 = 172 (120 x 78 x 172)
        # max pooling height floor(78/2) = 39 , max pooling width floor(172/2) = 86 (120 x 39 x 86)
        self.conv2 = nn.Conv2d(in_channels=120, out_channels=240, kernel_size=(2,3), stride=1)

        # convolution height (39-2+1)/1 = 38, convolution width (86-3+1)/1 = 84  (240 x 38 x 84)
        # max pooling height floor(38/2) = 19, max pooling width floor(84/2) = 42 (240 x 19 x 42)
        self.fc1 = nn.Linear(240*19*42, 5000)
        self.fc2 = nn.Linear(5000, 1200)
        self.fc31 = nn.Linear(1200, config["latent_vector_dim"])
        self.fc32 = nn.Linear(1200, config["latent_vector_dim"])

        # decode
        self.fc4 = nn.Linear(config["latent_vector_dim"], 1200)
        self.fc5 = nn.Linear(1200, 5000)
        self.fc6 = nn.Linear(5000, 240*19*42)

        # (240 x 19 x 42)
        # (240 x 38 x 84)
        self.conv3 = nn.ConvTranspose2d(in_channels=240, out_channels=120, kernel_size=(2,3), stride=1, output_padding=0)

        # (120 x 39 x 84)
        # (120 x 78 x 172)
        self.conv4 = nn.ConvTranspose2d(in_channels=120, out_channels=3, kernel_size=(5,7), stride=2, output_padding=1)
    
    def encode(self, x):
        # encode
        con1 = F.relu(self.conv1(x))
        max1 = F.max_pool2d(con1, 2, 2)
        con2 = F.relu(self.conv2(max1))
        max2 = F.max_pool2d(con2, 2, 2)
        lin0 = max2.view(-1,19*42*240)
        lin1 = F.relu(self.fc1(lin0))
        lin2 = F.relu(self.fc2(lin1))
        return self.fc31(lin2), self.fc32(lin2)

    def reparameterize(self, mu, logvar):
        """uses mean values and logvar values to sample random values"""
        # get standard deviation
        std = torch.exp(0.5*logvar)
        # print("STD: " + str(std))

        # returns a tensor of random numbers the shape of std with Mean = 0 and Variance = 1
        eps = torch.randn_like(std)
        # print("Rand: " + str(eps))

        # Return random number with Mean = mu and Variance = std^2
        return mu + eps*std


    def decode(self, z):
        # add two fully connected layers
        lin3 = F.relu(self.fc4(z))
        lin4 = F.relu(self.fc5(lin3))
        lin5 = F.relu(self.fc6(lin4))
        lin5 = lin5.reshape(-1, 240, 19, 42)
        max3 = F.interpolate(lin5, scale_factor=2)
        con3 = F.relu(self.conv3(max3))
        max4 = F.interpolate(con3, scale_factor=2)
        return torch.sigmoid(self.conv4(max4))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# save model to device and define optimizer as ADAM
print("save model to device and define optimizer as ADAM")
model = ConvVAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
print("done")

def loss_function(recon_x, x, mu, logvar, recon_weight=config["recon_weight"], kld_weight=config["kld_weight"]):
    """Since the posterior is Gaussian and the prior is Gaussian(0,I), there is a closed form for KL-Divergence
    See appendix B from VAE paper, https://arxiv.org/abs/1312.6114
    The normal KL Divergence is the sum of (recon_x * (recon_x/x).log())
    Using the reconstruction loss and KL Divergence helps for getting returns that are close yet unique.
    The weights of the  reconstruction + KL divergence losses"""
    # TO DO: Implement reconstruction + KL divergence losses summed over all elements and batch
    # for additional information on computing KL divergence
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    x = x.view(-1, 3*160*350)
    recon_x = recon_x.view(-1, 3*160*350)
    recon_loss = F.binary_cross_entropy(recon_x, x, size_average = False)
    KLD = 0.5 * ((-1 - logvar + mu.pow(2) + logvar.exp()).sum(axis = 0))
    # KLD = (recon_x * (recon_x/x).log())

    return recon_weight*recon_loss + kld_weight*KLD.mean()
   
def train(epoch):
    """This function trains the model. (Move epochs to within train and test functions)"""
    # switch model to training mode
    model.train()

    # initialize training loss
    train_loss = 0

    # iterate over each batch
    for batch_idx, data in enumerate(train_loader):

        # save batch to device
        data = data.to(device)

        # zero out the gradient for each iteration
        optimizer.zero_grad()

        # run forward method of VAE model
        recon_batch, mu, logvar = model(data)

        # calculated KL Divergence/Reconstruction Loss Function
        loss = loss_function(recon_batch, data, mu, logvar)

        # get gradient of loss function for each parameter
        loss.backward()

        # calculate total loss
        train_loss += loss.item()

        # perform step of backpropigation using ADAM optimization algo
        optimizer.step()

        # for every 10th epoch we print the training loss
        if batch_idx == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

    # print final loss
    if epoch % args.log_interval == 0:
        print('====> Epoch: {} Average loss: {:.4f}'.format(
              epoch, train_loss / len(train_loader.dataset)))


def test(epoch):
    # change model to evaluation mode
    model.eval()

    # initliaze test loss
    test_loss = 0

    # make sure no gradient is calculated
    with torch.no_grad():

        # iterate over each batch test data
        for i, data in enumerate(test_loader):
            # save batch to device
            data = data.to(device)

            # run forward method of VAE model
            recon_batch, mu, logvar = model(data)

            # calculate total loss
            test_loss += loss_function(recon_batch, data, mu, logvar).item()

            # if i == len(train_loader) and epoch % args.log_interval == 0:
            #     n = min(data.size(0), 8)
            #     comparison = torch.cat([data[:n],
            #                           recon_batch.view(args.batch_size, 1, 28, 28)[:n]])
            #     save_image(comparison.cpu(),
            #              'reconstructionKL_' + str(epoch) + '.png', nrow=n)

    # print out test loss
    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))

if __name__ == "__main__":
    print("Start")
    # Iterate over each epoch
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        test(epoch)