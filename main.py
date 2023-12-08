### IMPORTED LIBRARIES ########
#General libraries:
import numpy as np
import scipy as sp
import pandas as pd
import random
from PIL import Image
import os
import time
#import cv2 

#For Neural nets:
import itertools
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn.functional as F
from torch.autograd import Variable
from torch import autograd
from torchvision.utils import save_image

from numpy import linalg as LA
import scipy.sparse as spsparse
import scipy.linalg as splin

#For plotting:
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from src.dataset import MicroStructDatasetTruss
from src.utils import *
from src.gan_model import create_model
from src.physical_module import Estimator


# Dictionary storing network parameters.
params = {
    'save_fold':'savefold',
    'image_size':64,
    'batch_size': 16,# Batch size.
    'num_epochs': 500,# Number of epochs to train for.
    'learning_rate': 1e-4,# Learning rate.
    'beta1': 0.5,
    'beta2': 0.999,
    'save_epoch' : 25,
    'dim_embed': 128,
    'lr_fit': 1e-4,
    'coefD_pred':0.5,
    'coef_gencon': 1,
    'coefG_pred': 0.5,
    'num_z': 128,
    'num_con_c': 3,
    'ngpu': 1,
    'dataroot': 'image8',
    'workers': 0,
    'beta1': 0.5,
    'nz': 100,
    'ngf': 64,
    'ndf': 64,
    'l1_coef': 1000,
    'l2_coef': 5
    }

# system
dataroot = params['dataroot']
ngpu = params['ngpu']
## Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
savefold = params['save_fold']
workers = params['workers']
batch_size = params['batch_size']
beta1 = params['beta1']
n_con_c = 3
l1_coef = params['l1_coef']
l2_coef = params['l2_coef']

wd = os.getcwd()
save_models_folder = wd + "/" + savefold#'/output3/saved_models'
os.makedirs(save_models_folder, exist_ok=True)
os.makedirs(save_models_folder+"/save_image", exist_ok=True)

#-------------
image_size = params['image_size']
st = 1
ed = 2000
it = 1


print('Using device:', device)


#Additional Info when using cuda
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')
  


#######################################################################################
'''                                    Data loader                                 '''
#######################################################################################
#Define useful data transformation:
compose = transforms.Compose([ transforms.RandomCrop(image_size*3),
                               transforms.Resize(image_size),
                               transforms.Grayscale(num_output_channels=1), 
                               transforms.ToTensor(),
                               transforms.Normalize([0.5], [0.5]),
                            ])

# Create the Dataset:
dataset = MicroStructDatasetTruss( root_dir=dataroot, transform=compose, ed=ed)

# Create the Dataloader:
indices = list(range((len(dataset))))

train_sampler = SubsetRandomSampler(indices[st:ed:it])
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         num_workers=workers, sampler=train_sampler, pin_memory=True)



netG, discriminator, netD, netQ, netEstimator, netFitting = create_model(device)



coef_gencon = params['coef_gencon']
coefD_pred = params['coefD_pred']
coefG_pred = params['coefG_pred']

# Loss for discrimination between real and fake images.
criterionD = nn.BCELoss()
# Loss for discrete latent code.
criterionQ_dis = nn.CrossEntropyLoss()
# Loss for continuous latent code.
criterionQ_con = NormalNLLLoss()
l1_loss = nn.L1Loss()
l2_loss = nn.MSELoss()
# Adam optimiser is used.
optimdis = optim.Adam([{'params': discriminator.parameters()}],lr=params['learning_rate'], betas=(params['beta1'], params['beta2']))
optimD = optim.Adam([ {'params': netD.parameters()}], lr=params['learning_rate'], betas=(params['beta1'], params['beta2']))
optimG = optim.Adam([ {'params': netG.parameters()}], lr=params['learning_rate'], betas=(params['beta1'], params['beta2']))
optimQ = optim.Adam([ {'params': netQ.parameters()}], lr=params['learning_rate'], betas=(params['beta1'], params['beta2']))
optimEs = optim.Adam([ {'params': netEstimator.parameters()}], lr=2.5*params['learning_rate'], betas=(params['beta1'], params['beta2']))
optimFi = optim.SGD(netFitting.parameters(), lr = 10*params['learning_rate'], momentum=0.9)

# Fixed Noise
z = torch.randn(100, params['num_z'], 1, 1, device=device)
fixed_noise = z


if params['num_con_c'] == 2:
    #con_c = torch.rand(100, params['num_con_c'], 1, 1, device=device) * 2 - 1
    fixed_arrange = np.linspace(0, 1, 10)
    fixed_grid = np.meshgrid(fixed_arrange, fixed_arrange)
    fixed_grid = np.array(fixed_grid)
    fixed_grid = fixed_grid.reshape(2, 100)

    fixed_prop = np.zeros((100,2))
    #fixed_prop = fixed_prop.reshape(64, 2)
    #fixed_prop.transpose(1,0)
    for i in range(100):
        fixed_prop[i][0] = fixed_grid[0][i]
        fixed_prop[i][1] = fixed_grid[1][i]
    fixed_prop = (torch.from_numpy(fixed_prop)).type(torch.Tensor)
    fixed_prop = (fixed_prop).to(device)  
    fixed_c = fixed_prop.view(100, -1)


    
elif params['num_con_c'] == 1:
    fixed_prop = np.linspace(-1, 1, 100) 
    fixed_c = torch.from_numpy(fixed_prop).type(torch.Tensor).to(device).view(100, -1)

fixed_noise, fixed_c = noise_sample(params['num_con_c'], params['num_z'], 64, device)


# List variables to store results pf training.
img_list = []
G_losses = []
D_losses = []
iters = 0
print("-"*25)
print("Starting Training Loop...\n")
#print('Epochs: %d\n \nBatch Size: %d\n Length of Data Loader: %d') % (params['num_epochs'], params['batch_size'], len(dataloader)))
print("-"*25)

start_time = time.time()
one = torch.tensor(1, dtype=torch.float).to(device)
mone = one * -1

pred_fake_loss = one


for epoch in range(0, params['num_epochs']):
    epoch_start_time = time.time()

    for i, data in enumerate(dataloader, 0):
        # Get batch size
        b_size = data.size(0)
        # Transfer data tensor to GPU/CPU (device)
        real_ms = data.to(device)
        ################################## Discrimator training ################################
        # Updating discriminator and DHead
        optimD.zero_grad()
        optimdis.zero_grad()
        # Real data
        outputD_share_real = discriminator(real_ms)
        outputD_real = netD(outputD_share_real).view(-1)
        outputD_real = outputD_real.mean()
        outputD_real.backward(mone,retain_graph=True)

        # Fake data
        noise, c = noise_sample(params['num_con_c'], params['num_z'], b_size, device)
        fake_ms = netG(noise, c)
        output2 = discriminator(fake_ms.detach())
        outputD_fake = netD(output2).view(-1)
        outputD_fake = outputD_fake.mean()
        outputD_fake.backward(one,retain_graph=True)


        gradient_penalty = calculate_gradient_penalty(discriminator, netD, real_ms.data, fake_ms.data, b_size)
        gradient_penalty.backward()

        D_loss = outputD_real - outputD_fake - gradient_penalty

        optimdis.step()
        optimD.step()

        optimEs.zero_grad()
        ### update the estimatornet by real images
        # The loss of estimat of properties, (predict - fem)**2
        train_iter = 3
        if iters>100 and pred_real_loss<0.15:
            train_iter = 75
        if i % train_iter == 0 or i%20==0:
            pred_real = netEstimator(real_ms).view(-1, 3)
            fem_real = torch.from_numpy(Estimator(real_ms.cpu())).type(torch.Tensor).to(device).view(-1, 2)
            as_real = torch.from_numpy(calculate_special_area(real_ms.detach().cpu())).type(torch.Tensor).to(device).view(-1, 1)
            prop_real = torch.cat((fem_real, as_real), 1)
            pred_real_loss = l1_loss(pred_real, prop_real)
            pred_real_loss.backward()
            optimEs.step()
            
            
        ################################# Generator training #############
        if i%3 == 0:
            # Updating Generator and QHead
            optimG.zero_grad()  
            optimQ.zero_grad()
            # Update parameters.
            ################################## Fake image ################################
            # Fake data treated as real.
            fake_ms = netG(noise, c)
            outputG_share_fake = discriminator(fake_ms)
            outputG_fake = netD(outputG_share_fake).view(-1)
            errGen = outputG_fake.mean()
            errGen.backward(mone, retain_graph=True)


            if iters>2000:
                
                train_iter = 1
                if iters>100 and pred_fake_loss<0.15:
                    train_iter = 15
                if i % train_iter == 0 or i%20==0:
                    ##### The loss of estimator and FEM
                    pred_fake_detach = netEstimator(fake_ms.detach()).view(-1, 3)
                    fem_fake = torch.from_numpy(Estimator(fake_ms.detach().cpu())).type(torch.Tensor).to(device).view(-1,2)
                    as_fake = torch.from_numpy(calculate_special_area(fake_ms.detach().cpu())).type(torch.Tensor).to(device).view(-1,1)
                    prop_fake = torch.cat((fem_fake, as_fake), 1)
                    pred_fake_loss = l1_loss(pred_fake_detach, prop_fake)
                    optimEs.zero_grad()        
                    pred_fake_loss.backward()
                    optimEs.step()
                
                ## postior 
                q_mu, q_var = netQ(fake_ms)
                con_loss = criterionQ_con(c.view(-1, 3), q_mu.view(-1, 3), q_var.view(-1, 3))

                ##### The loss of estimator and FEM
                pred_fake = netEstimator(fake_ms).view(-1, 3)

                ### The loss of fitting and estimator
                #fitE = c2E(c, a1, a2, a3).view(-1)
                fitE = netFitting(c).view(-1, 3)
                fit_loss = l1_loss(pred_fake, fitE.detach())

                ### The loss of fitting and estimator
   
                G_loss =   l1_coef*(con_loss + l2_coef*fit_loss)                                                                                                  
                G_loss.backward()
                optimQ.step()
                # Calculate gradients.
            # Update parameters.
            optimG.step()

            #optimdis.step()

            # update the fitting net
            if iters>3000 :
                optimFi.zero_grad()  
                if i % train_iter == 0:
                    loss_fitting = l1_loss(fitE, prop_fake)
                else: loss_fitting = l1_loss(fitE, pred_fake.detach())
                loss_fitting.backward()
                optimFi.step()
                for p in netFitting.parameters():
                    p.data.clamp_(-10, 10)


        # Check progress of training.
        if i != 0 and i%10==0 and iters>3000:
            print('Loss of fitting\t', loss_fitting.item())
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\t  Loss of predict real images item:%.4f\t'
                  % (epoch+1, params['num_epochs'], i, len(dataloader), 
                    D_loss.item(), pred_real_loss.item()))
            print('[%d/%d][%d/%d]\t Loss_G: %.4f\t Loss of other items:%.4f\t, Loss of continuous item: %.4f\t Loss of predict fake images item:%.4f\t Loss of fitting item:%.4f'
                  % (epoch+1, params['num_epochs'], i, len(dataloader), 
                     errGen.item(), G_loss.item(), con_loss.item(), 
                    pred_fake_loss.item(),
                    fit_loss.item()))
        elif i != 0 and i%10==0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\t Loss_G: %.4f\t Loss of predict real:%.4f\t, Loss of predict fake:%.4f\t'
                  % (epoch+1, params['num_epochs'], i, len(dataloader), 
                    D_loss.item(), errGen.item(), pred_real_loss.item(),  pred_fake_loss.item()))             
        
        # Save the losses for plotting.
        G_losses.append(errGen.item())
        D_losses.append(D_loss.item())

        iters += 1


  
    epoch_time = time.time() - epoch_start_time
    print("Time taken for Epoch %d: %.2fs" %(epoch + 1, epoch_time))
    # Generate image after each epoch to check performance of the generator. Used for creating animated gif later.
    with torch.no_grad():
        gen_data = netG(fixed_noise, fixed_c).detach().cpu()
    img_list.append(vutils.make_grid(gen_data, nrow=10, padding=2, normalize=True))

    # Generate image to check performance of generator.
    if((epoch+1) == 1 or (epoch+1) == params['num_epochs']/2):
        with torch.no_grad():
            gen_data = netG(fixed_noise, fixed_c).detach().cpu()
        plt.figure(figsize=(10, 10))
        plt.axis("off")
        plt.imshow(np.transpose(vutils.make_grid(gen_data, nrow=10, padding=2, normalize=True), (1,2,0)))
        plt.savefig(savefold + "/Epoch_%d " %(epoch+1))
        plt.close('all')

    # Save network weights.
    if (epoch+1) % params['save_epoch'] == 0:
        torch.save({ \
            'netG' : netG.state_dict(),\
            'discriminator' : discriminator.state_dict(),\
            'netD' : netD.state_dict(),\
            'netQ' : netQ.state_dict(),\
            'netFitting': netFitting.state_dict(),
            'netEstimator': netEstimator.state_dict(),
            'optimD' : optimD.state_dict(),\
            'optimG' : optimG.state_dict(),\
            'params' : params}, savefold + '/model_epoch_%d' %(epoch+1))


training_time = time.time() - start_time
print("-"*50)
print('Training finished!\nTotal Time for Training: %.2fm' %(training_time / 60))
print("-"*50)
ww
# Save network weights.
torch.save({ 'netG' : netG.state_dict(),\
            'discriminator' : discriminator.state_dict(),\
            'netD' : netD.state_dict(),\
            'netQ' : netQ.state_dict(),\
            'netFitting': netFitting.state_dict(),
            'netEstimator': netEstimator.state_dict(),
            'optimD' : optimD.state_dict(),\
            'optimG' : optimG.state_dict(),\
            'params' : params}, savefold+'/final_model')






