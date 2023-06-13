import tensorflow as tf
import datetime
from torch.utils.tensorboard import SummaryWriter
import torchvision
import sys
import torch
import pandas as pd
import numpy as np
import gpytorch
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau
import matplotlib.pyplot as plt

from src.deep_gp import DeepGPRegression
import file_read

num_epochs = 400

output_dims = [5] * 1
num_inducing = 154
kernel_type = 'matern1.5'
num_samples = 3
lr = 0.04226454854439315


Train_X = file_read.Train_X
Train_y = file_read.Train_y
Val_X = file_read.Val_X
Val_y = file_read.Val_y
train_loader = file_read.train_loader
val_loader = file_read.val_loader

model = DeepGPRegression(train_x_shape=Train_X.shape, output_dims=output_dims,
                            num_inducing=num_inducing, kernel_type=kernel_type)

mll = gpytorch.mlls.DeepApproximateMLL(
    gpytorch.mlls.VariationalELBO(model.likelihood, model, Train_X.shape[-2]))

optimizer = torch.optim.AdamW([
    {'params': model.parameters()},
], lr=lr)

today = datetime.date.today()
today_str = today.strftime("%Y-%m-%d")
writer = SummaryWriter("logs/" + today_str + "_v1")

# training process
for epoch in range(num_epochs):
    epoch_loss = 0
    # set to train mode
    model.train()
    for batch, (X_batch, y_batch) in enumerate(train_loader):
        with gpytorch.settings.num_likelihood_samples(num_samples):
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

# plot of result
# evaluation
model.eval()
predictive_means, predictive_variances, test_lls = model.predict(val_loader)
pred_mean = predictive_means.mean(0)
pred_var = predictive_variances.mean(0)
test_lls = test_lls.mean(0)

rmse = torch.mean(torch.pow(predictive_means.mean(0) - Val_y, 2)).sqrt()
print(f"RMSE: {rmse.item()}, NLL: {-test_lls.mean().item()}")

sample_Val_y = Val_y[::20]
sample_pred = predictive_means.mean(0)[::20]

x_scatter = range(len(sample_Val_y))
idx_sorted = np.argsort(-sample_Val_y, axis=0)

plt.plot(x_scatter, sample_pred[idx_sorted], c='r')
plt.plot(x_scatter, sample_Val_y[idx_sorted], c='b')
plt.legend(['prediction','test RUL'])