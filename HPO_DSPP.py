import pandas as pd
import numpy as np
import optuna
import logging
import os
import sys
import time
import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, train_test_split
import torch
from torch.utils.data import TensorDataset, DataLoader
import gpytorch
import matplotlib.pyplot as plt
import plotly.io as pio
import gc

from src.utils.metric_utils import mse, mae, maximum_absolute_error, rmse
from src.utils.train_utils import EarlyStopping, EarlyStoppingWithModelSave
from src.dspp import DSPPRegression
import src.utils.file_utils as file_utils
import file_read


def objective_time_prune(trial, X, y, kf, epochs_inner, patience, time_tolerance=6000, metric=rmse):
    """
    Optuna trial. Performs n_splits CV to calculate a score for a given Hyperparameter Config

    Args:
        trial: Optuna trial
        X: features
        y: labels
        kf: Sklearn Kfold object
        epochs_inner: Number of inner epochs
        patience: Number of early stopping iterations
        time_tolerance:  Maximum number of seconds before cancel the trial
        metric: Evaluation metric to be optimized
        Q: Number of Quadrature points

    Returns:
        Average score of chosen metric of n_splits CV folds
    """
    # trial parameters
    lr = trial.suggest_float("lr", 1e-3, 1e-1, log=True)
    batch_size = trial.suggest_categorical("batch_size", [512, 1024])
    num_inducing = trial.suggest_int("num_inducing", 32, 500, log=True)
    num_quadrature_sites = trial.suggest_int("num_quadrature_sites", 5, 10)  # 5-10 recommended in gpytorch
    beta = trial.suggest_categorical("beta", [0.01, 0.05, 0.2, 1.0])  # follow search space of paper
    n_dspp_layers = trial.suggest_int("n_dspp_layers", 1, 3, log=True) # len(output_dims)
    n_dspp_out = trial.suggest_int("n_dspp_out", 1, 5, log=True)  # output_dims[i], log = True? 

    print(
        f'Current Configuration: n_dspp_layers:{n_dspp_layers} - n_dspp_out: {n_dspp_out} - num_inducing: {num_inducing} - '
        f'num_quadrature_sites: {num_quadrature_sites} - batch_size:{batch_size} - lr: {lr}')

    logging.info(
        f'Current Configuration: n_dspp_layers:{n_dspp_layers} - n_dspp_out: {n_dspp_out} - num_inducing: {num_inducing} - '
        f'num_quadrature_sites: {num_quadrature_sites} - batch_size:{batch_size} - lr: {lr}')

    # Model Evaluation on Train/Test Split
    scores=[]
    nc_split = 0
    for train_inner, val_inner in kf.split(X):
        nc_split = nc_split + 1

        print(f"{nc_split} split:")
        logging.info(f"{nc_split} split:")

        X_train_inner, X_val_inner, y_train_inner, y_val_inner = X[train_inner], X[val_inner], y[train_inner], y[val_inner]
        train_inner_dataset = TensorDataset(X_train_inner, y_train_inner)
        train_inner_loader = DataLoader(train_inner_dataset, batch_size=batch_size, shuffle=False)
        test_inner_dataset = TensorDataset(X_val_inner, y_val_inner)
        test_inner_loader = DataLoader(test_inner_dataset, batch_size=batch_size, shuffle=False)

        # initialize likelihood and model
        output_dims = [n_dspp_out] * n_dspp_layers # list obj
        model = DSPPRegression(train_x_shape=X_train_inner.shape, output_dims=output_dims,
                                num_inducing=num_inducing, Q=num_quadrature_sites)
        # Use the adam type optimizer
        optimizer = torch.optim.AdamW([
            {'params': model.parameters()},
        ], lr=lr)
        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.DeepPredictiveLogLikelihood(model.likelihood, model,
                                                        num_data=X_train_inner.size(0), beta=beta)

        start_time = time.time()
        # initialize early stopping
        early_stopping = EarlyStopping(patience=patience)

        for epoch in range(epochs_inner):
            # set Cholesky jitter
            gpytorch.settings.cholesky_jitter(1e-1)
            # set to train mode
            model.train()
            for batch, (X_batch, y_batch) in enumerate(train_inner_loader):
                optimizer.zero_grad()
                output = model(X_batch)
                loss = -mll(output, y_batch)
                loss.backward()
                # adjust learning weights
                optimizer.step()

            # Get into evaluation (predictive posterior) mode
            model.eval()
            # Make predictions
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                means, vars, ll = model.predict(test_inner_loader)
                weights = model.quad_weights.unsqueeze(-1).exp().cpu()
                score = metric((weights * means).sum(0), y_val_inner.cpu()).item()

            # Early Stopping
            early_stopping(score)

            if early_stopping.early_stop:
                print("Early stopping")
                logging.info("Early stopping")
                break

            if epoch % 10 == 0:
                cur_time = time.time()
                cur_duration = cur_time - start_time
                print(f"{epoch}/{epochs_inner} - Loss: {loss} - Score: {score} - Time: {cur_duration}")
                logging.info(f"{epoch}/{epochs_inner} - Loss: {loss} - Score: {score} - Time: {cur_duration}")

            # Pruner
            trial.report(score, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

            # Time based pruner
            train_time = time.time()
            if train_time - start_time > time_tolerance:
                print("Time Budget run out. Pruning Trial")
                logging.info("Time Budget run out. Pruning Trial")
                break

        # Set max epochs, as the maximum epoch over all inner splits
        # Max iteration reached
        if epoch == epochs_inner - 1:
            max_epochs = epochs_inner
        else:
            max_epochs = max(1, epoch - patience + 1)

        trial.set_user_attr("MAX_EPOCHS", int(max_epochs))

        # Have to take negative due to early stopping logic
        best_score = -early_stopping.best_score
        scores.append(best_score)

    return np.mean(scores)


def objective_train_test(trial, X, y, epochs_inner, patience, time_tolerance=1800, metric=rmse):
    """
    Optuna trial. Performs normal validation to calculate a score for a given Hyperparameter Config

    Args:
        trial: Optuna trial
        X: features
        y: labels
        epochs_inner: Number of inner epochs
        patience: Number of early stopping iterations
        time_tolerance:  Maximum number of seconds before cancel the trial
        metric: Evaluation metric to be optimized

    Returns:
        Average score of chosen metric of n_splits CV folds
    """
    # trial parameters
    lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
    batch_size = trial.suggest_categorical("batch_size", [512, 1024])
    num_inducing = trial.suggest_int("num_inducing", 50, 500, log=True)
    num_samples = trial.suggest_int("num_samples", 2, 10)
    kernel_type = trial.suggest_categorical("kernel_type", ["rbf", "matern0.5", "matern1.5"])
    n_gp_layers = trial.suggest_int("n_gp_layers", 1, 3, log=True) # len(output_dims)
    n_gp_out = trial.suggest_int("n_gp_out", 1, 10, log=True)  # output_dims[i], log = True? 

    print(
        f'Current Configuration: n_gp_layers:{n_gp_layers} - n_gp_out: {n_gp_out} - num_inducing: {num_inducing} - '
        f'num_samples: {num_samples} - batch_size:{batch_size} - num_inducing: {num_inducing} - lr: {lr}')
    logging.info(
        f'Current Configuration: n_gp_layers:{n_gp_layers} - n_gp_out: {n_gp_out} - num_inducing: {num_inducing} - '
        f'num_samples: {num_samples} - batch_size:{batch_size} - num_inducing: {num_inducing} - lr: {lr}')

    # Model Evaluation on Train/Test Split
    X_train_inner, X_val_inner, y_train_inner, y_val_inner = train_test_split(X, y, test_size=0.2, shuffle=False)

    # if torch.cuda.is_available():
    #     X_train_inner, X_val_inner, y_train_inner, y_val_inner = X_train_inner.cuda(), X_val_inner.cuda(), \
    #                                                              y_train_inner.cuda(), y_val_inner.cuda()

    train_inner_dataset = TensorDataset(X_train_inner, y_train_inner)
    train_inner_loader = DataLoader(train_inner_dataset, batch_size=batch_size, shuffle=False)

    test_inner_dataset = TensorDataset(X_val_inner, y_val_inner)
    test_inner_loader = DataLoader(test_inner_dataset, batch_size=batch_size, shuffle=False)

    # initialize likelihood and model
    # initialize model
    output_dims = [n_gp_out] * n_gp_layers # list obj
    model = DeepGPRegression(train_x_shape=X_train_inner.shape, output_dims=output_dims,
                             num_inducing=num_inducing, kernel_type=kernel_type)

    # if torch.cuda.is_available():
    #     model = model.cuda()

    # Use the adam type optimizer
    optimizer = torch.optim.AdamW([
        {'params': model.parameters()},
    ], lr=lr)

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.DeepApproximateMLL(
        gpytorch.mlls.VariationalELBO(model.likelihood, model, X_train_inner.shape[-2]))

    start_time = time.time()
    # initialize early stopping
    early_stopping = EarlyStopping(patience=patience)
    for epoch in range(epochs_inner):
        # set to train mode
        model.train()
        for batch, (X_batch, y_batch) in enumerate(train_inner_loader):
            with gpytorch.settings.num_likelihood_samples(num_samples):
                optimizer.zero_grad()
                output = model(X_batch)
                loss = -mll(output, y_batch)
                loss.backward()
                # adjust learning weights
                optimizer.step()

        # Get into evaluation (predictive posterior) mode
        model.eval()

        # Make predictions
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            predictions, predictive_variances, test_lls = model.predict(test_inner_loader)
            score = metric(predictions.mean(0), y_val_inner)

        # Early Stopping
        early_stopping(score)

        if early_stopping.early_stop:
            print("Early stopping")
            logging.info("Early stopping")
            break

        if epoch % 10 == 0:
            cur_time = time.time()
            cur_duration = cur_time - start_time
            print(f"{epoch}/{epochs_inner} - Loss: {loss} - Score: {score}")
            logging.info(f"{epoch}/{epochs_inner} - Loss: {loss} - Score: {score} - Time: {cur_duration}")

        # Pruner
        trial.report(score, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        # Time based pruner
        train_time = time.time()
        if train_time - start_time > time_tolerance:
            print("Time Budget run out. Pruning Trial")
            logging.info("Time Budget run out. Pruning Trial")
            break

    # Set max epochs, as the maximum epoch over all inner splits
    # Max iteration reached
    if epoch == epochs_inner - 1:
        max_epochs = epochs_inner
    # Early stopping
    else:
        max_epochs = max(1, epoch - patience + 1)

    # Memory Tracking
    logging.info("After model training")
    # logging.info(torch.cuda.memory_allocated() / 1024 ** 2)
    # logging.info(torch.cuda.memory_reserved() / 1024 ** 2)
    X_train_inner.detach()
    y_train_inner.detach()
    X_val_inner.detach()
    y_val_inner.detach()
    loss.detach()
    optimizer.zero_grad(set_to_none=True)
    mll.zero_grad(set_to_none=True)
    del X_train_inner, y_train_inner, X_val_inner, y_val_inner, train_inner_dataset, train_inner_loader,\
        test_inner_dataset, test_inner_loader, loss, optimizer, model, predictions, predictive_variances,\
        test_lls, output, mll
    gc.collect()
    # torch.cuda.empty_cache()
    logging.info("After memory clearing")
    # logging.info(torch.cuda.memory_allocated() / 1024 ** 2)
    # logging.info(torch.cuda.memory_reserved() / 1024 ** 2)

    trial.set_user_attr("MAX_EPOCHS", int(max_epochs))

    # Have to take negative due to early stopping logic
    best_score = -early_stopping.best_score

    return best_score


def HPO_DSPP(n_trials=50,
            epochs_inner=200,
            patience=10,
            file=None,
            pruner_type="None",
            time_tolerance=3600):

    dataset_name = 'FD001'

    Train_X = file_read.Train_X
    Train_y = file_read.Train_y

    kf = KFold(n_splits=4, shuffle=False)

    # File names
    today = datetime.date.today()
    today_str = today.strftime("%Y-%m-%d")
    now_time = time.strftime("%H:%M:%S", time.localtime())
    log_filename = "./log/HPO_DSPP_log_" + today_str + '_' + dataset_name + '_' + now_time + ".log"
    savedir = './data/experiments/DSPP'
    if not os.path.exists(savedir):
        os.mkdir(savedir)
    # study_infos_file = savedir + '/study_infos_DSPP_' + dataset_name + '.csv'
    nested_resampling_infos_file = savedir + '/nested_resampling_infos_DSPP_' + dataset_name + '.csv'
    # Python 3.6版本的logging模块中，basicConfig函数不支持encoding和force参数
    # logging.basicConfig(filename=log_filename, encoding='utf-8', level=logging.INFO, force=True)
    logging.basicConfig(filename=log_filename, filemode='w', level=logging.INFO)

    '''
    Commit comment on this new submitted task!!!!!!
    '''
    COMMIT = 'DSPP, WINDOW SIZE of 40, add history HPO plot, split=4' # Comment on what u have changed on this task being submitted
    logging.info(COMMIT)

    optuna.logging.enable_propagation() # Propagate logs to the root logger 'logging'

    if pruner_type == "HB":
        pruner = optuna.pruners.HyperbandPruner(min_resource=3, max_resource=epochs_inner, reduction_factor=3)
    elif pruner_type == "median":
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=20,
            interval_steps=5
        )
    else:
        pruner = optuna.pruners.NopPruner()

    # Create Study Obj For Optuna
    study_name = "DSPP " + today_str
    study = optuna.create_study(direction="minimize",
                                study_name=study_name,
                                pruner=pruner)

    # Suggest default parameters
    study.enqueue_trial({"n_dspp_layers": 1, 'n_dspp_out': 2, 'num_inducing': 128, 'batch_size': 1024,
                            'lr': 0.01, 'num_quadrature_sites': 8, 'beta': 0.05})

    study.optimize(lambda trial: objective_time_prune(trial, X=Train_X, y=Train_y,
                                                    kf=kf, 
                                                    epochs_inner=epochs_inner,
                                                    patience=patience,
                                                    time_tolerance=time_tolerance,
                                                    metric=rmse),
                                                    n_trials=n_trials)  # n_trials=N_TRIALS

    # plot HPO history visualization image
    fig_history = optuna.visualization.plot_optimization_history(study)
    fig_FI = optuna.visualization.plot_param_importances(study)
    fig_slice = optuna.visualization.plot_slice(study)
    PATH_OPTUNA_history = '/home/linyuhan/Dokumente/Masterarbeit/dataset/CMAPSS/code/DGP_Yuhan/images/optuna_HPO_DSPP_process/history_v1.png'
    PATH_OPTUNA_FI = '/home/linyuhan/Dokumente/Masterarbeit/dataset/CMAPSS/code/DGP_Yuhan/images/optuna_HPO_DSPP_process/FI_v1.png'
    PATH_OPTUNA_slice = '/home/linyuhan/Dokumente/Masterarbeit/dataset/CMAPSS/code/DGP_Yuhan/images/optuna_HPO_DSPP_process/slice_v1.png'

    pio.write_image(fig_history, PATH_OPTUNA_history)
    pio.write_image(fig_FI, PATH_OPTUNA_FI)
    pio.write_image(fig_slice, PATH_OPTUNA_slice)

    # # empty cuda cache to prevent memory issues
    # torch.cuda.empty_cache()

    # if torch.cuda.is_available():
    #     X_train_outer, X_test_outer, y_train_outer, y_test_outer = X_train_outer.cuda(), X_test_outer.cuda(), \
    #                                                                y_train_outer.cuda(), y_test_outer.cuda()
    
    # Append HPO Info
    study_info = study.trials_dataframe()
    study_info['dataset_name'] = dataset_name
    # study_infos = pd.concat([study_infos, study_info])
    # study_infos.to_csv(study_infos_file, index=False)

    # Refit with best trial
    best_trial = study.best_trial

    # best trial
    n_layers_best = best_trial.params['n_dspp_layers']
    n_out_best = best_trial.params['n_dspp_out']
    batch_size_best = best_trial.params['batch_size']
    lr_best = best_trial.params['lr']
    num_inducing_best = best_trial.params['num_inducing']
    num_quadrature_sites_best = best_trial.params['num_quadrature_sites']
    beta_best = best_trial.params['beta']

    # best parameters saving
    logging.info("Best trials: ")
    logging.info("Value: %s", best_trial.value)
    logging.info("params: ")
    for key, value in best_trial.params.items():
        logging.info("    {}: {}".format(key, value))

    # # Memory Tracking
    # logging.info("After final model training")
    # # logging.info(torch.cuda.memory_allocated() / 1024 ** 2)
    # # logging.info(torch.cuda.memory_reserved() / 1024 ** 2)
    # X_train_outer.detach()
    # X_test_outer.detach()
    # y_train_outer.detach()
    # y_test_outer.detach()
    # loss.detach()
    # optimizer.zero_grad(set_to_none=True)
    # mll.zero_grad(set_to_none=True)
    # del X_train_outer, X_test_outer, y_train_outer, y_test_outer, train_dataset, train_loader, test_dataset,\
    #     test_loader, loss, optimizer, model, predictions, predictive_variances, test_lls, output, mll
    # gc.collect()
    # # torch.cuda.empty_cache()
    # logging.info("After memory clearing")
    # # logging.info(torch.cuda.memory_allocated() / 1024 ** 2)
    # # logging.info(torch.cuda.memory_reserved() / 1024 ** 2)

if __name__ == "__main__":
    HPO_DSPP(n_trials=30,
            epochs_inner=200,
            patience=15,
            file="None",
            pruner_type="HB",
            time_tolerance=3600)

# 删除epochs_outer
# 删除n_split_inner 和 outer，只保留n_split