import tensorflow as tf
import datetime
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import TensorDataset, DataLoader
import tensorboard as tf
import torchvision
import sys
import torch
import pandas as pd
import numpy as np
import gpytorch
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau
import matplotlib.pyplot as plt

from src.dspp import DSPPRegression
import file_read

num_epochs = 100
BATCH_SIZE = 512
output_dims = [4] * 3
num_inducing = 71
num_quadrature_sites = 7
beta = 0.01
lr = 0.03776271737103609

Train_X = file_read.Train_X
Train_y = file_read.Train_y
train_loader = file_read.train_loader

## tensorboard settings ##
today = datetime.date.today()
today_str = today.strftime("%Y-%m-%d")
# tensorboard_logdir = "./tensorboard_logs/" + "FD001" + today_str + "_NoWindow"
tensorboard_logdir = "./tensorboard_logs/" + "_DSPP_" + f'{file_read.Num_Slurm_HPO}'


writer = SummaryWriter(tensorboard_logdir)

## model initialize ##
model = DSPPRegression(train_x_shape=Train_X.shape, output_dims=output_dims,
                        num_inducing=num_inducing, Q=num_quadrature_sites)

mll = gpytorch.mlls.DeepPredictiveLogLikelihood(model.likelihood, model,
                                                num_data=Train_X.size(0), beta=beta)

optimizer = torch.optim.AdamW([
    {'params': model.parameters()},
], lr=lr)

# training process
for epoch in range(num_epochs):
    # set Cholesky jitter
    gpytorch.settings.cholesky_jitter(1e-1)
    epoch_loss = 0
    # set to train mode
    model.train()
    for batch, (X_batch, y_batch) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(X_batch)
        loss = -mll(output, y_batch)
        writer.add_scalar('loss', loss, epoch)
        loss.backward()
        # adjust learning weights
        optimizer.step()
        epoch_loss += loss.item()

    avg_epoch_loss = epoch_loss / len(train_loader)
    print("Epoch: {} - Loss: {:.4f}".format(epoch, avg_epoch_loss))
writer.close()

torch.save(model.state_dict(), file_read.model_state_file_name)