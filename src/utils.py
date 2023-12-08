### IMPORTED LIBRARIES ########
#General libraries:
import numpy as np
import scipy as sp
import random
from PIL import Image
import os


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch import autograd

device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
def contours_peri(image):
    image = (image.numpy().transpose(1, 2, 0)+1)/2. * 255.0
    image = np.array(image, dtype='uint8')
    _, thresh = cv2.threshold(image, 127, 255, 0)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    #print(contours.shape)
    lengths = []
    areas = []
    for i in range(len(contours)):
        length = cv2.arcLength(contours[i], True)
        lengths.append(length)
        #area = cv2.contourArea(contours[i])
        #areas.append(area)
    len_sum = sum(lengths)
    #area_sum = sum(areas)
    special_area = len_sum/1300#(area_sum+1e6)
    return special_area

def calculate_special_area(image_batch):
    sa_batch = np.zeros(len(image_batch))
    for i in range(len(image_batch)):
        sa_batch[i] = contours_peri(image_batch[i])
    return sa_batch*2


def weights_init(m):
    if(type(m) == nn.ConvTranspose2d or type(m) == nn.Conv2d):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif(type(m) == nn.BatchNorm2d):
        if m.bias is not None:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)


def noise_sample( n_con_c, n_z, batch_size, device):
    noise = torch.randn(batch_size, n_z, 1, 1, device=device)
    # Random uniform between 0 and 1.
    con_c = torch.rand(batch_size, n_con_c, device=device)
    return noise, con_c

class NormalNLLLoss:
    def __call__(self, x, mu, var):
        logli = -0.5 * (var.mul(2 * np.pi) + 1e-6).log() - (x - mu).pow(2).div(var.mul(2.0) + 1e-6)
        nll = -(logli.sum(1).mean())

        return nll


def calculate_gradient_penalty(discriminator, netD, real_images, fake_images, batch_size):
    lambda_term = 10
    eta = torch.FloatTensor(batch_size,1,1,1).uniform_(0,1)
    eta = eta.expand(batch_size, real_images.size(1), real_images.size(2), real_images.size(3))
    eta = eta.to(device)
    interpolated = eta * real_images + ((1 - eta) * fake_images)
    interpolated = interpolated.to(device)
    # define it to calculate gradient
    interpolated = Variable(interpolated, requires_grad=True)
    # calculate probability of interpolated examples
    prob_interpolated = netD(discriminator(interpolated))
    # calculate gradients of probabilities with respect to examples
    gradients = autograd.grad(outputs=prob_interpolated, inputs=interpolated,
                           grad_outputs=torch.ones(
                               prob_interpolated.size()).to(device),
                           create_graph=True, retain_graph=True)[0]
    grad_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lambda_term
    return grad_penalty
