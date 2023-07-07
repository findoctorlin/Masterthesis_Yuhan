import file_read
from src.dspp import DSPPRegression

import torch
import gpytorch
import matplotlib.pyplot as plt
import numpy as np

Train_X = file_read.Train_X
Train_y = file_read.Train_y
Test_X = file_read.Test_X
Test_y = file_read.Test_y
train_loader = file_read.train_loader
test_loader = file_read.test_loader

# initialization of hpyerparameters
num_epochs = 100
BATCH_SIZE = 512
output_dims = [4] * 3
num_inducing = 71
num_quadrature_sites = 7
beta = 0.01
lr = 0.03776271737103609

model_after = DSPPRegression(train_x_shape=Train_X.shape, output_dims=output_dims,
                        num_inducing=num_inducing, Q=num_quadrature_sites)
model_state_after = torch.load(file_read.model_state_path)
model_after.load_state_dict(model_state_after)

# evaluation
model_after.eval()
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    predictive_means, predictive_variances, test_lls = model_after.predict(test_loader)
    weights = model_after.quad_weights.unsqueeze(-1).exp().cpu()
    # `predictive_means` currently contains the predictive output from each Gaussian in the mixture.
    # To get the total mean output, we take a weighted sum of these means over the quadrature weights.
    pred_mean = (weights * predictive_means).sum(0)
    pred_var = (weights * predictive_variances).sum(0)
    rmse = (pred_mean - Test_y.cpu()).pow(2.0).mean().sqrt().item()
    NLL = test_lls.mean().item()
    

print(f"RMSE: {rmse}, NLL: {NLL}")

def plot_mean(Test_y, pred_mean, sample_ratio):
    sample_Test_y = Test_y[::sample_ratio]
    sample_pred = pred_mean[::sample_ratio]

    x_scatter = range(len(sample_Test_y))
    idx_sorted = np.argsort(-sample_Test_y, axis=0)

    plt.plot(x_scatter, sample_pred[idx_sorted], c='r')
    plt.plot(x_scatter, sample_Test_y[idx_sorted], c='b')
    plt.title(f'RMSE={rmse.item():.4f}, NLL={-test_lls.mean().item():.4f}')
    plt.legend(['prediction','test RUL'])
    plt.show()

def plot_corvariance(Test_y, pred_mean, pred_var, sample_ratio):
    sample_Test_y = Test_y[::sample_ratio]
    sample_pred = pred_mean[::sample_ratio]
    sample_pred_var = pred_var[::sample_ratio]

    x_scatter = range(len(sample_Test_y))
    idx_sorted = np.argsort(-sample_Test_y, axis=0)

    lower = sample_pred - sample_pred_var.sqrt() * 2.576
    upper = sample_pred + sample_pred_var.sqrt() * 2.576

    plt.plot(x_scatter, sample_Test_y[idx_sorted], c='b')
    plt.plot(x_scatter, sample_pred[idx_sorted], c='r')
    plt.fill_between(x_scatter, lower[idx_sorted].detach().cpu().numpy(), upper[idx_sorted].detach().cpu().numpy(), alpha=0.5)
    plt.title(f'RMSE={rmse.item():.4f}, NLL={-test_lls.mean().item():.4f}')

    plt.show()

def plot_corvariance_engine(Test_y, pred_mean, pred_var, test_lls, loc_start, loc_end, id_engine):
    sample_Test_y = Test_y[loc_start:loc_end+1]
    sample_pred = pred_mean[loc_start:loc_end+1]
    sample_pred_var = pred_var[loc_start:loc_end+1]

    x_scatter = range(len(sample_Test_y))
    idx_sorted = np.argsort(-sample_Test_y, axis=0)

    lower = sample_pred - sample_pred_var.sqrt() * 2.576
    upper = sample_pred + sample_pred_var.sqrt() * 2.576

    # plt.plot(x_scatter, sample_Test_y[idx_sorted], c='b')
    # plt.plot(x_scatter, sample_pred[idx_sorted], c='r')
    plt.scatter(x_scatter, sample_Test_y[idx_sorted], c='b', s=10)
    plt.scatter(x_scatter, sample_pred[idx_sorted], c='r', s=10)
    plt.fill_between(x_scatter, lower[idx_sorted].detach().cpu().numpy(), upper[idx_sorted].detach().cpu().numpy(), alpha=0.5)
    
    rmse_engine = torch.mean(torch.pow(pred_mean[loc_start:loc_end+1] - Test_y[loc_start:loc_end+1], 2)).sqrt()
    NLL_engine = -test_lls[loc_start:loc_end+1].mean()

    plt.title(f'engine {id_engine}: RMSE={rmse_engine.item():.4f}, NLL={NLL_engine:.4f}')
    plt.show()

# 68 test engine [0,67], number 67 is last one, need manually plot
def get_group_index(data):
    result_dict = {0: [0, 0]}
    start_index = 0
    end_index = 0
    num_group = 0
    for i in range(len(data)):
        if i == 3531:
            break
        if data[i+1] < data[start_index]:
            continue
        else:
            end_index = i
            result_dict[num_group] = [start_index, end_index]
            start_index = i+1
            num_group += 1
    return result_dict

# plot_mean(Test_y, pred_mean, 10)
# plot_corvariance(Test_y, pred_mean, pred_var, 30)

group_dict = get_group_index(Test_y)
plot_corvariance_engine(Test_y, pred_mean, pred_var, test_lls, 3446, 3531, 67)
for id_engine in range(0,67):
    plot_corvariance_engine(Test_y, pred_mean, pred_var, test_lls, group_dict[id_engine][0], group_dict[id_engine][1], id_engine)