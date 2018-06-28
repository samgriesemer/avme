# Sam Griesemer 04/15/17

import torch
import torchvision.transforms as transforms
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import torchvision

from models.ame import AME
from models.vae import VAE

import math
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline

# initialize hyperparams
lr = 1.0
bsz = 100
clip = 0.5
log_interval = 10
criterion = nn.BCELoss()
sample_iters = 10
ame_list = []
vae_list = []
loss_list = []
epochs = 21

# get data
transform=transforms.Compose([transforms.ToTensor()
                             ])
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=bsz, shuffle=True, num_workers=2)

testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=bsz, shuffle=False, num_workers=2)

data_list = []
onehot_labels = []
for i, data in enumerate(trainloader):
    inputs, labels = data
    inputs = torch.round(inputs)
    onehot = torch.zeros(bsz,10)
    for j in range(bsz):
        onehot[j][int(labels[j])] = 1
    onehot_labels.append(onehot.view(1,bsz,10))
    data_list.append(inputs.view(1,bsz,784))
data_list = torch.cat(data_list)
onehot_labels = torch.cat(onehot_labels)

ninp = data_list.size(-1)
mid_size = 400
latent_size = 80

ame_enc_v = AME(ninp*3+10, mid_size, mid_size, latent_size)
ame_dec_v = AME(latent_size+10, mid_size, mid_size, ninp)

criterion = nn.BCELoss()
criterion.size_average = False

def clip_gradient(model, clip):
    """Computes a gradient clipping coefficient based on gradient norm."""
    totalnorm = 0
    for p in model.parameters():
        modulenorm = p.grad.data.norm()
        totalnorm += modulenorm ** 2
    totalnorm = math.sqrt(totalnorm)
    return min(1, clip / (totalnorm + 1e-6))

def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)

def loss_function(recon_x, x, mu, logvar):
    BCE = criterion(recon_x, x)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5)

    return BCE + KLD

def train(epoch):
    
    # iterator variables
    count = 1
    total_loss = 0

    # loop through training data
    for i in range(len(data_list)):
        
        batch = Variable(data_list[i])
        label = Variable(onehot_labels[i])

        canvas = Variable(torch.zeros(bsz,784))
        
        output_v = Variable(torch.zeros(bsz,784))
        
        # instantiate encoder and decoder network layers
        h_enc_v_1 = (Variable(torch.zeros(bsz,mid_size)), Variable(torch.zeros(bsz,mid_size)))
        h_enc_v_2 = (Variable(torch.zeros(bsz,latent_size)), Variable(torch.zeros(bsz,latent_size)))
        h_enc_v_3 = (Variable(torch.zeros(bsz,latent_size)), Variable(torch.zeros(bsz,latent_size)))
        
        h_dec_v_1 = (Variable(torch.zeros(bsz,mid_size)), Variable(torch.zeros(bsz,mid_size)))
        h_dec_v_2 = (Variable(torch.zeros(bsz,ninp)), Variable(torch.zeros(bsz,ninp)))
        
        # zero gradients and reset KLD each data batch iteration
        ame_enc_v.zero_grad()
        ame_dec_v.zero_grad()
        KLD = 0
        
        # begin loop through K specified generative iterations
        for t in range(sample_iters):
            
            # push batch through encoder and store hidden states
            mu_v, logvar_v, h_enc_v_1, h_enc_v_2, h_enc_v_3 = ame_enc_v(torch.cat([batch,batch.sub(canvas.sigmoid()),output_v,label],1), h_enc_v_1, h_enc_v_2, enc=h_enc_v_3)
            
            # compute params of latent space
            std_v = logvar_v.mul(0.5).exp_()
            eps_v = torch.FloatTensor(std_v.size()).normal_()
            eps_v = Variable(eps_v)
            sample_v = eps_v.mul(std_v).add_(mu_v)
            
            # compute KL divergence 
            KLD_v_elem = mu_v.pow(2).add_(logvar_v.exp()).mul_(-1).add_(1).add_(logvar_v)
            KLD += torch.sum(KLD_v_elem).mul_(-0.5)
            
            # decode latent space sample and map back to sample space
            output_v, h_dec_v_1, h_dec_v_2 = ame_dec_v(torch.cat([sample_v,label],1), h_dec_v_1, h_dec_v_2)
            
            # add decoder output to recurring canvas element
            canvas = output_v.add(canvas)
            
        canvas.sigmoid_()
        
        # compute binary cross entropy loss and add KLD component
        loss = criterion(canvas, batch)
        loss += KLD
        loss.backward()

        # clip encoder gradients
        clipped_lr = lr * clip_gradient(ame_enc_v, clip)
        for p in ame_enc_v.parameters():
            p.data.add_(-clipped_lr, p.grad.data)
            
        # clip decoder gradients
        clipped_lr = lr * clip_gradient(ame_dec_v, clip)
        for p in ame_dec_v.parameters():
            p.data.add_(-clipped_lr, p.grad.data)
        
        # store loss data (mainly for viz)
        loss_list.append(loss.data[0])
        total_loss += loss.data[0]
        
        # print loss stats
        if count % 10 == 0:
            avg_loss = total_loss / count
            print('Epoch:', epoch, 'Iter:', count, 'Avg Loss:', avg_loss, 'Cur Loss:', loss.data[0])
        count += 1

def test(epoch):
    vae.eval()
    test_loss = 0
    for data, _ in testloader:
        data = Variable(data, volatile=True)
        recon_batch, mu, logvar = vae(data)
        test_loss += loss_function(recon_batch, data, mu, logvar).data[0]

    test_loss /= len(testloader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))


for epoch in range(1, epochs+1):
    train(epoch)
