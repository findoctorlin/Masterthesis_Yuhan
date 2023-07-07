import file_read
from src.deep_gp import DeepGPRegression

import torch
import matplotlib.pyplot as plt
import numpy as np

Train_X = file_read.Train_X
Train_y = file_read.Train_y
Test_X = file_read.Test_X
Test_y = file_read.Test_y
train_loader = file_read.train_loader
test_loader = file_read.test_loader

### find the clipped unit ###
df_test_raw_RUL = file_read.df_test_raw_RUL
df_unit_RUL = df_test_raw_RUL[['unit_number', 'RUL']].copy()
selected_units = []
max_unit_number = df_test_raw_RUL['unit_number'].max()
for unit in range(1, max_unit_number+1):
    unit_df = df_unit_RUL[df_unit_RUL['unit_number'] == unit]
    min_rul = unit_df['RUL'].min()
    if min_rul > 105:
        selected_units.append(unit)

all_units = list(range(1,max_unit_number+1))
clipped_units = [x for x in all_units if x not in selected_units] # remaining unit's number

### initialization of hpyerparameters ###
output_dims = [3] * 3
num_inducing = 55
kernel_type = 'matern0.5'

model_after = DeepGPRegression(train_x_shape=Train_X.shape, output_dims=output_dims,
                            num_inducing=num_inducing, kernel_type=kernel_type)
model_state_after = torch.load(file_read.model_state_path)
model_after.load_state_dict(model_state_after)

# evaluation
model_after.eval()
predictive_means, predictive_variances, test_lls = model_after.predict(test_loader)
pred_mean = predictive_means.mean(0)
pred_var = predictive_variances.mean(0)
test_lls = test_lls.mean(0)

rmse = torch.mean(torch.pow(predictive_means.mean(0) - Test_y, 2)).sqrt()
print(f"RMSE: {rmse.item()}, NLL: {-test_lls.mean().item()}")

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

    plt.plot(x_scatter, sample_Test_y[idx_sorted], c='b')
    plt.plot(x_scatter, sample_pred[idx_sorted], c='r')
    # plt.scatter(x_scatter, sample_Test_y[idx_sorted], c='b', s=10)
    # plt.scatter(x_scatter, sample_pred[idx_sorted], c='r', s=10)
    plt.fill_between(x_scatter, lower[idx_sorted].detach().cpu().numpy(), upper[idx_sorted].detach().cpu().numpy(), alpha=0.5)
    
    rmse_engine = torch.mean(torch.pow(pred_mean[loc_start:loc_end+1] - Test_y[loc_start:loc_end+1], 2)).sqrt()
    NLL_engine = -test_lls[loc_start:loc_end+1].mean()

    plt.title(f'engine {id_engine}: RMSE={rmse_engine.item():.4f}, NLL={NLL_engine:.4f}')
    plt.show()

# 68 test engine [0,67], number 67 is last one
def get_group_index(data):
    result_dict = {}
    start_index = 0
    end_index = 0
    num_group = 0
    for i in range(len(data)):
        if i == 0:
            continue
        if i == len(data)-1:
            end_index = i
            result_dict[num_group] = [start_index, end_index]
        if data[i] < 105:
            continue
        else:
            end_index = i-1
            result_dict[num_group] = [start_index, end_index]
            start_index = i
            num_group += 1
    return result_dict

# plot_mean(Test_y, pred_mean, 10)
# plot_corvariance(Test_y, pred_mean, pred_var, 30)

result_dict = get_group_index(Test_y)
group_dict = {key: result_dict[value] for key, value in zip(clipped_units, result_dict.keys())}

for id_engine in clipped_units:
    plot_corvariance_engine(Test_y, pred_mean, pred_var, test_lls, group_dict[id_engine][0], group_dict[id_engine][1], id_engine)