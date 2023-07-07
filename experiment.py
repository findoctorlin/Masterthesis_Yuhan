from HPO_DGP import HPO_DGP
from HPO_DSPP import HPO_DSPP

if __name__ == "__main__":
    # config
    n_trials = 50  # Number of HPO trials to conduct per Outer CV split
    epochs_inner = 500  # Number of inner epochs
    patience = 11  # Number of early stopping iterations

    # hpo = input("Enter HPO Study type: ")
    hpo = "DSPP"
    # file = input("Enter file: ") # eg. 'CMAPSS'
    file = "CMAPSS"
    if file == "None":
        file = 'CMAPSS'

    # Optuna Pruner Type, either "None" or "HB" for Hyperband
    # pruner_type = input("Pruner Type: ")
    pruner_type = "HB"
    # Maximum number of seconds before cancel a HPO trial
    # time_tolerance = int(input("Enter time tolerance: "))
    time_tolerance = 6000

    if hpo == "DGP":
        HPO_DGP(n_trials=n_trials,
            epochs_inner=epochs_inner,
            patience=patience,
            file=file,
            pruner_type=pruner_type,
            time_tolerance=time_tolerance)
        
    elif hpo == "DSPP":
        HPO_DSPP(n_trials=n_trials,
            epochs_inner=epochs_inner,
            patience=patience,
            file=file,
            pruner_type=pruner_type,
            time_tolerance=time_tolerance)