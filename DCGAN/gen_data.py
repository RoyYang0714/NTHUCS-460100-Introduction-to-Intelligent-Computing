import os, random
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from argparse import ArgumentParser

parser = ArgumentParser()

parser.add_argument("-d", dest="data_type", help="datset type", default="0")

args = parser.parse_args()

# Set random seed for reproducibility
manualSeed = 999

random.seed(manualSeed)
torch.manual_seed(manualSeed)

# dataset
dataroot = "./Type_{}".format(args.data_type)   # Root directory for dataset
workers = 2                                     # Number of workers for dataloader

# hyperparemeter
batch_size = 128                                # Batch size during training
image_size = 64                                 # image size
nc = 3                                          # Number of channels in images
nz = 100                                        # Size of z latent vector
ngf = 64                                        # Size of feature maps in generator
ndf = 64                                        # Size of feature maps in discriminator
num_epochs = 100                                # Number of training epochs
lr = 3e-4                                       # Learning rate for optimizers
beta1 = 0.5                                     # Beta1 hyperparam for Adam optimizers
ngpu = 1                                        # Number of GPUs available. Use 0 for CPU mode.

# number of fake image
if args.data_type == '1':
    fake_num = 1800
else:
    fake_num = 1900

# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

# Generator Code
class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)

class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)

netG = Generator(ngpu).to(device)
netG.load_state_dict(torch.load('./G{}.pkl'.format(args.data_type)))
netD = Discriminator(ngpu).to(device)
netD.load_state_dict(torch.load('./D{}.pkl'.format(args.data_type)))

# Plot the real images
fixed_noise = torch.randn(fake_num, nz, 1, 1, device=device)
img_list = []

fake = np.array(netG(fixed_noise).detach().cpu()) * 127.5 + 127.5
fake = fake.astype(int)

for idx in tqdm(range(len(fake)) ,position=0, leave=True):
    plt.axis("off")
    plt.imshow(np.transpose(fake[idx], (1,2,0)))
    plt.savefig('./Type_{}/Fake/{}.jpg'.format(args.data_type, idx+1))
    plt.close()
