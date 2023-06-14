import file_read
import train_DGP
from src.deep_gp import DeepGPRegression

import torch
import matplotlib.pyplot as plt
import numpy as np

Train_X = file_read.Train_X
Train_y = file_read.Train_y
Val_X = file_read.Val_X
Val_y = file_read.Val_y
train_loader = file_read.train_loader
val_loader = file_read.val_loader

output_dims = train_DGP.output_dims
num_inducing = train_DGP.num_inducing
kernel_type = train_DGP.kernel_type
num_samples = train_DGP.num_samples
lr = 0.04226454854439315

model_after = DeepGPRegression(train_x_shape=Train_X.shape, output_dims=output_dims,
                            num_inducing=num_inducing, kernel_type=kernel_type)
model_state_after = torch.load('/home/linyuhan/Dokumente/Masterarbeit/dataset/CMAPSS/code/DGP_Yuhan/model_state_after.pth')
model_after.load_state_dict(model_state_after)

# evaluation and plot
model_after.eval()
predictive_means, predictive_variances, test_lls = model_after.predict(val_loader)
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
plt.show()