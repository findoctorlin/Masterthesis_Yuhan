import file_read
from src.deep_gp import DeepGPRegression

import torch
import matplotlib.pyplot as plt
import numpy as np

Train_X = file_read.Train_X
Train_y = file_read.Train_y
Val_X = file_read.Val_X
Val_y = file_read.Val_y
Test_X = file_read.Test_X
Test_y = file_read.Test_y
train_loader = file_read.train_loader
val_loader = file_read.val_loader
test_loader = file_read.test_loader

# initialization of hpyerparameters
output_dims = [10] * 1
num_inducing = 128
kernel_type = 'matern1.5'

model_after = DeepGPRegression(train_x_shape=Train_X.shape, output_dims=output_dims,
                            num_inducing=num_inducing, kernel_type=kernel_type)
model_state_after = torch.load(file_read.model_state_path)
model_after.load_state_dict(model_state_after)

# evaluation
model_after.eval()
predictive_means, predictive_variances, test_lls = model_after.predict(val_loader)
pred_mean = predictive_means.mean(0)
pred_var = predictive_variances.mean(0)
test_lls = test_lls.mean(0)

rmse = torch.mean(torch.pow(predictive_means.mean(0) - Val_y, 2)).sqrt()
print(f"RMSE: {rmse.item()}, NLL: {-test_lls.mean().item()}")

def plot_mean(Val_y, pred_mean, sample_ratio):
    sample_Val_y = Val_y[::sample_ratio]
    sample_pred = pred_mean[::sample_ratio]

    x_scatter = range(len(sample_Val_y))
    idx_sorted = np.argsort(-sample_Val_y, axis=0)

    plt.plot(x_scatter, sample_pred[idx_sorted], c='r')
    plt.plot(x_scatter, sample_Val_y[idx_sorted], c='b')
    plt.title(f'RMSE={rmse.item():.4f}, NLL={-test_lls.mean().item():.4f}')
    plt.legend(['prediction','test RUL'])
    plt.show()

def plot_corvariance(Val_y, pred_mean, pred_var, sample_ratio):
    sample_Val_y = Val_y[::sample_ratio]
    sample_pred = pred_mean[::sample_ratio]
    sample_pred_var = pred_var[::sample_ratio]

    x_scatter = range(len(sample_Val_y))
    idx_sorted = np.argsort(-sample_Val_y, axis=0)

    lower = sample_pred - sample_pred_var.sqrt() * 2.576
    upper = sample_pred + sample_pred_var.sqrt() * 2.576

    plt.plot(x_scatter, sample_Val_y[idx_sorted], c='b')
    plt.plot(x_scatter, sample_pred[idx_sorted], c='r')
    plt.fill_between(x_scatter, lower[idx_sorted].detach().cpu().numpy(), upper[idx_sorted].detach().cpu().numpy(), alpha=0.5)
    plt.title(f'RMSE={rmse.item():.4f}, NLL={-test_lls.mean().item():.4f}')

    plt.show()


# plot_mean(Val_y, pred_mean, 10)
plot_corvariance(Val_y, pred_mean, pred_var, 10)
## plot of uncertainty