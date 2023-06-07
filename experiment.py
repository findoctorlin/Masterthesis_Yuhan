from HPO_DGP import HPO_DGP

if __name__ == "__main__":
    # config
    n_splits = 5  # Number of Outer CV Splits
    n_trials = 50  # Number of HPO trials to conduct per Outer CV split
    epochs_inner = 400   # Number of inner epochs
    patience = 5  # Number of early stopping iterations
    epochs_outer = 500  # Number of Epochs to evaluate the best configuration
    patience_outer = 10  # Number of early stopping iterations

    hpo = input("Enter HPO Study type: ")
    file = input("Enter file: ") # eg. 'CMAPSS'
    if file == "None":
        file = 'CMAPSS'

    # Optuna Pruner Type, either "None" or "HB" for Hyperband
    pruner_type = input("Pruner Type: ")
    # Maximum number of seconds before cancel a HPO trial
    time_tolerance = int(input("Enter time tolerance: "))

    if hpo == "DGP":
        HPO_DGP(n_trials=n_trials,
            epochs_inner=epochs_inner,
            patience=patience,
            file=file,
            pruner_type=pruner_type,
            time_tolerance=time_tolerance)