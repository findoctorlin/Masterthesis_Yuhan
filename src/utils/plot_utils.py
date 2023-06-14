import matplotlib.pyplot as plt
import numpy as np

predictive_means, predictive_variances, test_lls = model.predict(val_loader)
pred_mean = predictive_means.mean(0)
pred_var = predictive_variances.mean(0)
test_lls = test_lls.mean(0)

rmse = torch.mean(torch.pow(predictive_means.mean(0) - Val_y, 2)).sqrt()
print(f"RMSE: {rmse.item()}, NLL: {-test_lls.mean().item()}")

def plot_final_mean(Val_y, pred_mean, sample_ratio):
    sample_Val_y = Val_y[:sample_ratio]
    sample_pred_mean = pred_mean[:sample_ratio]
    x_scatter = range(len(sample_Val_y))
    idx_sorted = np.argsort(-sample_Val_y, axis=0)

    plt.plot(x_scatter, sample_pred_mean[idx_sorted], c='r')
    plt.plot(x_scatter, sample_Val_y[idx_sorted], c='b')
    plt.legend(['prediction','test RUL'])