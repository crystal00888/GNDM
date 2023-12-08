#General libraries:
import numpy as np
import random
import time

#For Neural nets:
import itertools
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torchvision.utils as vutils
import torch.nn.functional as F
from torch.autograd import Variable
from torch import autograd
from src.utils import weights_init



class Concat_embed(nn.Module):

    def __init__(self, embed_dim, projected_embed_dim):
        super(Concat_embed, self).__init__()
        self.projection = nn.Sequential(
            nn.Linear(in_features=embed_dim, out_features=projected_embed_dim),
            nn.BatchNorm1d(num_features=projected_embed_dim),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
            )
    def forward(self, inp, embed):
        projected_embed = self.projection(embed)
        replicated_embed = projected_embed.repeat(1, 1, 1, 1).permute(2,  3, 0, 1)
        hidden_concat = torch.cat([inp, replicated_embed], 1)

        return hidden_concat

class Embedding_net(nn.Module):
    def __init__(self, input_dim, dim_embed ):
        super(Embedding_net, self).__init__()
        self.input_dim = input_dim
        self.fc_embedding = nn.Sequential(
            nn.Linear(input_dim, dim_embed),
            nn.BatchNorm1d(dim_embed),
            #nn.GroupNorm(8, self.dim_embed),
            nn.LeakyReLU(),

            nn.Linear(dim_embed, dim_embed),
            nn.BatchNorm1d(dim_embed),
            #nn.GroupNorm(8, self.dim_embed),
            nn.LeakyReLU(),

            nn.Linear(dim_embed, dim_embed),
            nn.LeakyReLU()
        )

    def forward(self, x):
        x = x.view(-1, self.input_dim) + 1e-8
        return self.fc_embedding(x)

class ConditionalBatch_Norm2d(nn.Module):
    def __init__(self, num_features, embed_dim):
        super().__init__()
        self.num_features = num_features
        self.bn = nn.BatchNorm2d(num_features, affine=False)

        # self.embed = nn.Linear(dim_embed, num_features * 2, bias=False)
        # self.embed.weight.data[:, :num_features].normal_(1, 0.02)  # Initialise scale at N(1, 0.02)
        # self.embed.weight.data[:, num_features:].zero_()  # Initialise bias at 0
        # # self.embed = spectral_norm(self.embed) #seems not work

        self.embed_gamma = nn.Linear(embed_dim, num_features, bias=False)
        self.embed_beta = nn.Linear(embed_dim, num_features, bias=False)

    def forward(self, x, y):
        out = self.bn(x)

        # gamma, beta = self.embed(y).chunk(2, 1)
        # out = gamma.view(-1, self.num_features, 1, 1) * out + beta.view(-1, self.num_features, 1, 1)
        gamma = self.embed_gamma(y).view(-1, self.num_features, 1, 1)
        beta = self.embed_beta(y).view(-1, self.num_features, 1, 1)
        out = out + out*gamma + beta

        return out
        
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.image_size = 64
        self.num_channels = 1
        self.noise_dim = 128
        self.embed_dim = 16
        self.projected_embed_dim = 30
        self.latent_dim = self.noise_dim + self.projected_embed_dim
        self.ngf = 64
        self.embed1 = Embedding_net(1, self.embed_dim)
        self.embed2 = Embedding_net(1, self.embed_dim)
        

        self.netG1 = nn.Sequential(
            nn.ConvTranspose2d(self.noise_dim, self.ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(self.ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(self.ngf * 8, self.ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf * 4),
            nn.ReLU(True),)
        
        self.netG2 = nn.Sequential(           
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(self.ngf * 4 , self.ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf * 2),
            nn.ReLU(True),
        )
        self.netG3 = nn.Sequential(
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(self.ngf * 2, self.ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(self.ngf, self.num_channels, 4, 2, 1, bias=False),
            nn.Tanh()
             # state size. (num_channels) x 64 x 64
            )
        self.condbn1 = ConditionalBatch_Norm2d(self.noise_dim, self.embed_dim)
        self.condbn2 = ConditionalBatch_Norm2d(self.ngf * 4, 2)
        self.condbn3 = ConditionalBatch_Norm2d(self.ngf * 2, self.embed_dim)
    def forward(self, z, c):
        embed_vector1 = self.embed1(c[:, 0].view(-1, 1))
        embed_vector2 = self.embed2(c[:, 1].view(-1, 1))
        c_cos = torch.cos(c[:, 2]*torch.pi).view(-1, 1)
        c_sin = torch.sin(c[:, 2]*torch.pi).view(-1, 1)
        #embed_vector1 = c[:, 0].view(-1, 1)
        #embed_vector2 = c[:, 1].view(-1, 1)
        #z_c = torch.cat((z.view(-1, self.noise_dim),), dim=1)
        lat_vec1 = self.condbn1(z, embed_vector2)
        x1 = self.netG1(lat_vec1)                                        
        x2 = self.condbn2(x1, torch.cat((c_cos, c_sin), 1))
        x3 = self.netG2(x2)
        x4 = self.condbn3(x3, embed_vector1)
        x5 = self.netG3(x4)
        return x5


class SeqDiscriminator(nn.Module):
    def __init__(self):
        super(SeqDiscriminator, self).__init__()
        self.image_size = 64
        self.num_channels = 1
        self.embed_dim = 128
        self.projected_embed_dim = 30
        self.ndf = 64
        self.B_dim = 128
        self.C_dim = 16

        self.netD_1 = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(self.num_channels, self.ndf, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(self.ndf),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(self.ndf, self.ndf * 2, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(self.ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(self.ndf * 2, self.ndf * 4, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(self.ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(self.ndf * 4 , self.ndf * 8, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(self.ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )
    def forward(self, x):
        x = self.netD_1(x)
        return x


class DHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.ndf = 64
        self.netD_1 = nn.Sequential(
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(self.ndf*8, self.ndf, 4, 1, 0),
            nn.LeakyReLU(0.2, inplace=True),
            )
        self.linear1 = nn.Sequential(nn.Linear(self.ndf, self.ndf),
                        nn.ReLU(),
                        nn.Linear(self.ndf, 1)) 

    def forward(self, x):
        output = self.netD_1(x)
        output = self.linear1(output.view(-1,self.ndf))
        return output
        
#------------------------------------------------------------------------------
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet_embed(nn.Module):
    def __init__(self, block, num_blocks, nc=1 ):
        super(ResNet_embed, self).__init__()
        self.in_planes = 64

        self.main = nn.Sequential(
            nn.Conv2d(nc, 64, kernel_size=3, stride=1, padding=1, bias=False),  # h=h
            # nn.Conv2d(nc, 64, kernel_size=4, stride=2, padding=1, bias=False),  # h=h/2
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2,2), #h=h/2 64
            # self._make_layer(block, 64, num_blocks[0], stride=1),  # h=h
            self._make_layer(block, 64, num_blocks[0], stride=2),  # h=h/2 32
            self._make_layer(block, 128, num_blocks[1], stride=2), # h=h/2 16
            self._make_layer(block, 256, num_blocks[2], stride=2), # h=h/2 8
            self._make_layer(block, 512, num_blocks[3], stride=2), # h=h/2 4
            # nn.AvgPool2d(kernel_size=4)
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.linear = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 3, bias=True),
        )


    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        features = self.main(x)
        features = features.view(features.size(0), -1)
        out = self.linear(features)
        return out


class QNet(nn.Module):
    def __init__(self):
        super(QNet, self).__init__()
        self.image_size = 64
        self.num_channels = 1
        self.ndf = 16


        self.netQ1 = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(self.num_channels, self.ndf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(self.ndf*4, self.ndf * 2, 4, 2, 1, bias=False),
        )
        
        self.netQ2 = nn.Sequential(
            nn.BatchNorm2d(self.ndf * 2),
            nn.LeakyReLU(0.2, inplace=False),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(self.ndf * 2, self.ndf * 4, 4, 2, 1, bias=False),
            #nn.BatchNorm2d(self.ndf * 4),
            #nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            #nn.Conv2d(self.ndf * 4 , self.ndf * 8, 4, 2, 1, bias=False),      
        )
        self.netQ3 = nn.Sequential(
            #nn.BatchNorm2d(self.ndf * 2),
            #nn.LeakyReLU(0.2, inplace=False),
            # state size. (ndf*2) x 16 x 16
            #nn.Conv2d(self.ndf * 2, self.ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(self.ndf * 4 , self.ndf * 8, 4, 2, 1, bias=False),      
        )
        self.conv_mu1 = nn.Linear(self.ndf*2*16*16, 1, bias=True)
        self.conv_var1 = nn.Linear(self.ndf*2*16*16, 1, bias=True)
        
        self.conv_mu2 = nn.Linear(self.ndf*4*8*8, 1, bias=True)
        self.conv_var2 = nn.Linear(self.ndf*4*8*8, 1, bias=True)
    
        self.conv_mu3 = nn.Linear(self.ndf*8*4*4, 1, bias=True)
        self.conv_var3 = nn.Linear(self.ndf*8*4*4, 1, bias=True)
        
    def forward(self, x):
        x1 = self.netQ1(x)
        mu1 = self.conv_mu1(x1.view(-1, self.ndf*2*16*16))
        var1 = torch.exp(self.conv_var1(x1.view(-1, self.ndf*2*16*16)))
        
        x2 = self.netQ2(x1)
        mu2 = self.conv_mu2(x2.view(-1, self.ndf*4*8*8))
        var2 = torch.exp(self.conv_var2(x2.view(-1, self.ndf*4*8*8)))

        x3 = self.netQ3(x2)
        mu3 = self.conv_mu3(x3.view(-1, self.ndf*8*4*4))
        var3 =torch.exp(self.conv_var3(x3.view(-1, self.ndf*8*4*4)))
        
        mu = torch.cat((mu1, mu2, mu3), 1)
        var = torch.cat((var1, var2, var3), 1)
        return mu, var



def NetEstimator():
    return ResNet_embed(BasicBlock, [2,2,2,2])

class Fitting(nn.Module):
    def __init__(self):
        super(Fitting, self).__init__()
        self.input = 4
        self.output = 2
        
        self.linear = nn.Linear(self.input, self.output, bias=True)
        self.linear_sin = nn.Linear(1, 1, bias=True)
    def forward(self, x):
        x = x.view(-1, 3)
        x1 = x[:, 0].view(-1,1)
        x2 = (x[:, 0]**2).view(-1, 1)

        y1 = x[:, 1].view(-1, 1)
        y2 = (x[:, 1]**2).view(-1, 1)
        
        z1 = self.linear_sin(x[:, 2].view(-1, 1))
        #z2 = (torch.sin(x[:, 2]*2*torch.pi)**2).view(-1, 1)
        
        x_input = torch.cat((x1, x2, y1, y2), dim=1)
        output1 = self.linear(x_input)
        output2 = (torch.cos(2*torch.pi*z1)).view(-1, 1)
        output = torch.cat((output2, output1), 1)
        return output.view(-1, 3)



def create_model(device):
    # Initialise the network.
    netG = Generator().to(device)
    netG.apply(weights_init)
    print(netG)
    pytorch_total_params = sum(p.numel() for p in netG.parameters() if p.requires_grad)
    print("*****************Total parameters to optimize: ", pytorch_total_params)

    discriminator = SeqDiscriminator().to(device)
    discriminator.apply(weights_init)
    print(discriminator)
    pytorch_total_params = sum(p.numel() for p in discriminator.parameters() if p.requires_grad)
    print("*****************Total parameters to optimize: ", pytorch_total_params)

    netD = DHead().to(device)
    netD.apply(weights_init)
    print(netD)
    pytorch_total_params = sum(p.numel() for p in netD.parameters() if p.requires_grad)
    print("*****************Total parameters to optimize: ", pytorch_total_params)

    netQ = QNet().to(device)
    netQ.apply(weights_init)
    print(netQ)
    pytorch_total_params = sum(p.numel() for p in netQ.parameters() if p.requires_grad)
    print("*****************Total parameters to optimize: ", pytorch_total_params)

    netEstimator = NetEstimator().to(device)
    netEstimator.apply(weights_init)
    print(netEstimator)
    pytorch_total_params = sum(p.numel() for p in netEstimator.parameters() if p.requires_grad)
    print("*****************Total parameters to optimize: ", pytorch_total_params)

    netFitting = Fitting().to(device)
    netFitting.apply(weights_init)
    print(netFitting)
    pytorch_total_params = sum(p.numel() for p in netFitting.parameters() if p.requires_grad)
    print("*****************Total parameters to optimize: ", pytorch_total_params)
    return netG, discriminator, netD, netQ, netEstimator, netFitting