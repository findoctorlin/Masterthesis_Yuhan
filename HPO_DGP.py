import pandas as pd
import numpy as np
import optuna
import openml
import logging
import os
import sys
import time

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, train_test_split

import torch
from torch.utils.data import TensorDataset, DataLoader
import gpytorch

from src.utils.metric_utils import mse, mae, maximum_absolute_error
from src.utils.train_utils import EarlyStopping, EarlyStoppingWithModelSave

from src.deep_gp import DeepGPRegressionGP

import gc


def objective_time_prune(trial, X, y, kf,
                         epochs_inner, patience,
                         time_tolerance=1800,
                         metric=mse):
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

    Returns:
        Average score of chosen metric of n_splits CV folds
    """
    # trial parameters
    lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
    kernel_type = trial.suggest_categorical("kernel_type", ["rbf", "matern0.5", "matern1.5"])
    batch_size = trial.suggest_categorical("batch_size", [256, 512, 1024])
    num_inducing = trial.suggest_int("num_inducing", 50, 1000, log=True)
    num_samples = trial.suggest_int("num_samples", 2, 15)
    n_gp_layers = trial.suggest_int("n_gp_layers", 1, 5, log=True) # len(output_dims)
    n_gp_out = trial.suggest_int("n_gp_out", 1, 4, log=True)  # log = True?

    print(
        f'Current Configuration: n_gp_layers:{n_gp_layers} - n_gp_out: {n_gp_out} - num_inducing: {num_inducing} - '
        f'num_samples: {num_samples} - kernel_type: {kernel_type} - lr: {lr}')
    logging.info(
        f'Current Configuration: n_gp_layers:{n_gp_layers} - n_gp_out: {n_gp_out} - num_inducing: {num_inducing} - \
        num_samples: {num_samples} - kernel_type: {kernel_type} - lr: {lr}')

    start_time = time.time()
    early_stopping = EarlyStopping(patience=patience)
    
    # Inner Nested Resampling Split
    for epoch in range(epochs_inner):
        cur_split_nr = 1
        scores = []
        for train_inner, val_inner in kf_inner.split(X): # 改动kf->kf_inner
            # Temp model name for this cv split to load and save
            temp_model_name = './models/temp/DGP_temp_split_' + str(cur_split_nr) + '.pt'

            # Train/Val Split
            X_train_inner, X_val_inner, y_train_inner, y_val_inner = X[train_inner], X[val_inner], y[train_inner], y[
                val_inner]

            if torch.cuda.is_available():
                X_train_inner, X_val_inner, y_train_inner, y_val_inner = X_train_inner.cuda(), X_val_inner.cuda(), \
                                                                         y_train_inner.cuda(), y_val_inner.cuda()

            train_inner_dataset = TensorDataset(X_train_inner, y_train_inner)
            train_inner_loader = DataLoader(train_inner_dataset, batch_size=batch_size, shuffle=False)

            test_inner_dataset = TensorDataset(X_val_inner, y_val_inner)
            test_inner_loader = DataLoader(test_inner_dataset, batch_size=batch_size, shuffle=False)

            # initialize likelihood and model
            # initialize model
            output_dims = [n_gp_out] * n_gp_layers
            model = DeepGPRegression(train_x_shape=X_train_inner.shape, output_dims=output_dims,
                                     num_inducing=num_inducing, kernel_type=kernel_type)

            # Load existing model if not the first epoch
            if epoch != 0:
                state_dict = torch.load(temp_model_name)
                model.load_state_dict(state_dict)

            if torch.cuda.is_available():
                model = model.cuda()

            # Use the adam optimizer
            optimizer = torch.optim.AdamW([
                {'params': model.parameters()},
            ], lr=lr)

            # "Loss" for GPs - the marginal log likelihood
            mll = gpytorch.mlls.DeepApproximateMLL(
                gpytorch.mlls.VariationalELBO(model.likelihood, model, X_train_inner.shape[-2]))

            # Train Model for 1 epochs
            for i in range(1):
                # set to train mode
                model.train()
                for batch, (X_batch, y_batch) in enumerate(train_inner_loader):
                    with gpytorch.settings.num_likelihood_samples(num_samples):
                        # Zero gradient
                        optimizer.zero_grad()
                        # Output from model
                        output = model(X_batch)
                        # Calc loss and backprop gradients
                        loss = -mll(output, y_batch)
                        loss.backward()
                        # adjust learning weights
                        optimizer.step()

            # Get into evaluation (predictive posterior) mode
            model.eval()

            # Make predictions by feeding model through likelihood
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                predictions, predictive_variances, test_lls = model.predict(test_inner_loader)
                score = metric(predictions.mean(0), y_val_inner)

            # Append Model Scores
            scores.append(score.cpu().item())

            # Save model
            torch.save(model.state_dict(), temp_model_name)
            # Increase cur_split_nr by 1
            cur_split_nr += 1

            # Clear cache
            del X_train_inner, y_train_inner, X_val_inner, y_val_inner, loss, optimizer, model,\
                predictions, predictive_variances, test_lls, output, mll
            gc.collect()
            torch.cuda.empty_cache()

        # average Scores
        average_scores = np.mean(scores)

        # Early Stopping
        early_stopping(average_scores)
        if early_stopping.early_stop:
            print("Early stopping")
            break

        # Time based pruner
        train_time = time.time()
        if train_time - start_time > time_tolerance:
            print("Time Budget run out. Pruning Trial")
            break

        print(f"{epoch}/{epochs_inner} - Score: {average_scores}")

    cur_split_nr = 1
    for _, _ in kf_inner.split(X):
        # Temp model name for this cv split to load and save
        temp_model_name = './models/temp/DGP_temp_split_' + str(cur_split_nr) + '.pt'
        os.remove(temp_model_name)
        cur_split_nr += 1

    # Memory Tracking
    logging.info("After model training")
    logging.info(torch.cuda.memory_allocated() / 1024 ** 2)
    logging.info(torch.cuda.memory_reserved() / 1024 ** 2)

    # Set max epochs
    # Max iteration reached
    if epoch == epochs_inner - 1:
        max_epochs = epochs_inner
    # Early stopping
    else:
        max_epochs = max(1, epoch - patience + 1)
    trial.set_user_attr("MAX_EPOCHS", int(max_epochs))

    # Have to take negative due to early stopping logic
    best_score = -early_stopping.best_score

    return best_score

def objective_train_test(trial, X, y, epochs_inner, patience, time_tolerance=1800, metric=mse):
    """
    Optuna trial. Performs n_splits CV to calculate a score for a given Hyperparameter Config

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
    batch_size = trial.suggest_categorical("batch_size", [512, 1024, 2048])
    num_inducing = trial.suggest_int("num_inducing", 50, 800, log=True)
    num_samples = trial.suggest_int("num_samples", 2, 15)
    kernel_type = trial.suggest_categorical("kernel_type", ["rbf", "matern0.5", "matern1.5"])
    n_gp_layers = trial.suggest_int("n_gp_layers", 1, 4, log=True) # len(output_dims)
    n_gp_out = trial.suggest_int("n_gp_out", 1, 4, log=True)  # log = True?

    print(
        f'Current Configuration: n_gp_layers:{n_gp_layers} - n_gp_out: {n_gp_out} - num_inducing: {num_inducing} - '
        f'num_samples: {num_samples} - num_inducing: {num_inducing} - lr: {lr}')

    # Model Evaluation on Train/Test Split
    X_train_inner, X_val_inner, y_train_inner, y_val_inner = train_test_split(X, y, test_size=0.25, shuffle=False)

    if torch.cuda.is_available():
        X_train_inner, X_val_inner, y_train_inner, y_val_inner = X_train_inner.cuda(), X_val_inner.cuda(), \
                                                                 y_train_inner.cuda(), y_val_inner.cuda()

    train_inner_dataset = TensorDataset(X_train_inner, y_train_inner)
    train_inner_loader = DataLoader(train_inner_dataset, batch_size=batch_size, shuffle=False)

    test_inner_dataset = TensorDataset(X_val_inner, y_val_inner)
    test_inner_loader = DataLoader(test_inner_dataset, batch_size=batch_size, shuffle=False)

    # initialize likelihood and model
    # initialize model
    output_dims = [n_gp_out] * n_gp_layers
    model = DeepGPRegression(train_x_shape=X_train_inner.shape, output_dims=output_dims,
                             num_inducing=num_inducing, kernel_type=kernel_type)

    if torch.cuda.is_available():
        model = model.cuda()

    # Use the adam optimizer
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
            break

        if epoch % 10 == 0:
            print(f"{epoch}/{epochs_inner} - Loss: {loss} - Score: {score}")

        # Pruner
        trial.report(score, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

        # Time based pruner
        train_time = time.time()
        if train_time - start_time > time_tolerance:
            print("Time Budget run out. Pruning Trial")
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
    logging.info(torch.cuda.memory_allocated() / 1024 ** 2)
    logging.info(torch.cuda.memory_reserved() / 1024 ** 2)
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
    torch.cuda.empty_cache()
    logging.info("After memory clearing")
    logging.info(torch.cuda.memory_allocated() / 1024 ** 2)
    logging.info(torch.cuda.memory_reserved() / 1024 ** 2)

    trial.set_user_attr("MAX_EPOCHS", int(max_epochs))

    best_score = -early_stopping.best_score

    return best_score

def HPO_DGP(n_splits=5,
            n_trials=30,
            epochs_inner=500,
            patience=10,
            file=None,
            PPV=True,
            dataset_id=216,
            pruner_type="None",
            time_tolerance=1800,
            patience_outer=20):
    if file == "CMAPSS":
        file_path = '../../train_CMAPSS.csv'
        df = pd.read_csv(file_path, sep=',', index_col=0)

    # prepare data for training
    X = df.iloc[:, 0:14]
    y = df.iloc[:, -1]
    dataset_name = 'CMAPSS'

    # File names
    log_filename = "./log/HPO_DGP_log_" + dataset_name + ".log"
    savedir = './data/experiments/DGP'
    if not os.path.exists(savedir):
        os.mkdir(savedir)
    study_infos_file = savedir + '/study_infos_DGP_' + dataset_name + '.csv'
    nested_resampling_infos_file = savedir + '/nested_resampling_infos_DGP_' + dataset_name + '.csv'
    logging.basicConfig(filename=log_filename, encoding='utf-8', level=logging.INFO, force=True)
    optuna.logging.enable_propagation()

    kf = KFold(n_splits=n_splits, shuffle=False) # 改动

    # data preprocess
    for train_outer, test_outer in kf_outer.split(np.array(X)):
        X_train_outer, X_test_outer, y_train_outer, y_test_outer = X[train_outer], X[test_outer], y[train_outer], y[test_outer]
        # scaler applied to features and labels
        scaler = StandardScaler()
        X_train_outer = scaler.fit_transform(X_train_outer)
        X_test_outer = scaler.transform(X_test_outer)
        # y_scaler = MinMaxScaler() # 改动: 没有应用y_scaler
        # y_train_outer = y_scaler.fit_transform(y_train_outer.reshape(-1, 1)).flatten()
        X_train_outer = torch.from_numpy(X_train_outer).float()
        X_test_outer = torch.from_numpy(X_test_outer).float()

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
        study_name = "CV_DGP " + str(n_splits)
        study = optuna.create_study(direction="minimize",
                                    study_name=study_name,
                                    pruner=pruner)

        # Suggest default parameters
        study.enqueue_trial({"n_gp_layers": 2, 'n_gp_out': 1, 'num_inducing': 128, 'batch_size': 1024,
                             'lr': 0.01, 'num_samples': 10, 'kernel_type': 'rbf'})
        try:
            if len(X_train_outer) > 10000: # 改动100000
                study.optimize(lambda trial: objective_train_test(trial, X=X_train_outer, y=y_train_outer,
                                                                  epochs_inner=epochs_inner,
                                                                  time_tolerance=time_tolerance, patience=patience),
                                                                  n_trials=n_trials)  # n_trials=N_TRIALS
            else:
                study.optimize(lambda trial: objective_time_prune(trial, X=X_train_outer, y=y_train_outer, kf=kf,
                                                                  epochs_inner=epochs_inner,
                                                                  time_tolerance=time_tolerance, patience=patience),
                                                                  n_trials=n_trials) # n_trials=N_TRIALS
        except:  # most likely runtime error due to not enough memory
            logging.info(sys.exc_info()[0], "occurred.")
            logging.info("Aborting Study")
        # empty cuda cache to prevent memory issues
        torch.cuda.empty_cache()

        if torch.cuda.is_available():
            X_train_outer, X_test_outer, y_train_outer, y_test_outer = X_train_outer.cuda(), X_test_outer.cuda(), \
                                                                       y_train_outer.cuda(), y_test_outer.cuda()
        
        # Append HPO Info
        study_info = study.trials_dataframe()
        study_info['dataset_name'] = dataset_name
        study_infos = pd.concat([study_infos, study_info])
        study_infos.to_csv(study_infos_file, index=False)

        # Refit with best trial
        # best trial
        n_layers_best = best_trial.params['n_gp_layers']
        n_out_best = best_trial.params['n_gp_out']
        batch_size_best = best_trial.params['batch_size']
        lr_best = best_trial.params['lr']
        num_inducing_best = best_trial.params['num_inducing']
        num_samples_best = best_trial.params['num_samples']
        kernel_type_best = best_trial.params['kernel_type']
        # # As max epochs inferred through smaller dataset, increase it by 10%
        # max_epochs = int(best_trial.user_attrs['MAX_EPOCHS'] * 1.1)

        # Build dataloader for train and test dataset
        train_dataset = TensorDataset(X_train_outer, y_train_outer)
        train_loader = DataLoader(train_dataset, batch_size=batch_size_best, shuffle=False)

        test_dataset = TensorDataset(X_test_outer, y_test_outer)
        test_loader = DataLoader(test_dataset, batch_size=batch_size_best, shuffle=False)

        # initialize model
        output_dims = [n_out_best] * n_layers_best
        model = DeepGPRegression(train_x_shape=X_train_outer.shape, output_dims=output_dims,
                                 num_inducing=num_inducing_best, kernel_type=kernel_type_best)

        if torch.cuda.is_available():
            model = model.cuda()

        # Use the adam optimizer
        optimizer = torch.optim.AdamW([
            {'params': model.parameters()},
        ], lr=lr_best)

        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.DeepApproximateMLL(
            gpytorch.mlls.VariationalELBO(model.likelihood, model, X_train_outer.shape[-2]))
        early_stopping = EarlyStoppingWithModelSave(patience=patience_outer)
        # time training loop
        start = time.time()
        for epoch in range(epochs_outer):
            model.train()
            epoch_loss = []
            for batch, (X_batch, y_batch) in enumerate(train_loader):
                with gpytorch.settings.num_likelihood_samples(num_samples_best):
                    optimizer.zero_grad()
                    output = model(X_batch)
                    loss = -mll(output, y_batch)
                    loss.backward()
                    optimizer.step()
                    epoch_loss.append(loss.item())

            epoch_loss = sum(epoch_loss) / len(epoch_loss)

            if epoch % 5 == 0:
                print(f"{epoch}/{epochs_outer} - loss: {epoch_loss}")

            early_stopping(epoch_loss, model)
            if early_stopping.early_stop:
                print("Early stopping")
                break
        end = time.time()

        # Predictions
        model.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            predictions, predictive_variances, test_lls = model.predict(test_loader)

        predictions_orig = torch.from_numpy(
            y_scaler.inverse_transform(predictions.mean(0).cpu().reshape(-1, 1)).flatten()).float()

        mse_score = mse(y_test_outer.cpu(), predictions_orig).item()
        mae_score = mae(y_test_outer.cpu(), predictions_orig).item()
        max_score = maximum_absolute_error(y_test_outer.cpu(), predictions_orig).item()

        # NLL Var(Y)=Var(aX+b)=a^2*Var(X)
        var = predictive_variances.mean(0).cpu()
        y_min = y_scaler.data_min_
        y_max = y_scaler.data_max_
        var_transformed = var * (y_max - y_min) ** 2
        mean_transformed = predictions_orig
        nll = torch.nn.GaussianNLLLoss()
        nll_score = nll(mean_transformed, y_test_outer.cpu(), var_transformed).item()
        train_time = end - start

        nested_resampling_info = pd.DataFrame({'dataset_name': [dataset_name],
                                               'n_layers': [n_layers_best],
                                               'n_out': [n_out_best], 'batch_size': [batch_size_best],
                                               'lr': [lr_best], 'num_inducing': [num_inducing_best],
                                               'num_samples': [num_samples_best], 'kernel_type': [kernel_type_best],
                                               'MAX_EPOCHS': [epoch], 'mse': [mse_score],
                                               'mae': [mae_score], 'max_absolute_error': [max_score],
                                               'nll_score': [nll_score], 'train_time': [train_time]})
        nested_resampling_infos = pd.concat([nested_resampling_infos, nested_resampling_info])
        # split_nr += 1
        nested_resampling_infos.to_csv(nested_resampling_infos_file, index=False)

        os.remove("checkpoint.pt")

        # Memory Tracking
        logging.info("After final model training")
        logging.info(torch.cuda.memory_allocated() / 1024 ** 2)
        logging.info(torch.cuda.memory_reserved() / 1024 ** 2)
        X_train_outer.detach()
        X_test_outer.detach()
        y_train_outer.detach()
        y_test_outer.detach()
        loss.detach()
        optimizer.zero_grad(set_to_none=True)
        mll.zero_grad(set_to_none=True)
        del X_train_outer, X_test_outer, y_train_outer, y_test_outer, train_dataset, train_loader, test_dataset,\
            test_loader, loss, optimizer, model, predictions, predictive_variances, test_lls, output, mll
        gc.collect()
        torch.cuda.empty_cache()
        logging.info("After memory clearing")
        logging.info(torch.cuda.memory_allocated() / 1024 ** 2)
        logging.info(torch.cuda.memory_reserved() / 1024 ** 2)

if __name__ == "__main__":
    HPO_DGP(n_splits=5,
            n_trials=50,
            epochs_inner=400,
            patience=5,
            file="None",
            PPV=True,
            pruner_type="HB",
            time_tolerance=3600,
            patience_outer=10)

# 删除epochs_inner
# 删除n_split_inner 和 outer，只保留n_split