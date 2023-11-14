# Deep Gaussian Process implementation

# Requiements

python 3.6

Requiements:

- gpytorch==1.6.0
- keras==2.6.0
- Keras-Preprocessing==1.1.2
- matplotlib==2.2.5
- numpy==1.17.3
- optuna==3.0.6
- pandas==1.1.5
- plotly==5.15.0
- pyparsing==2.4.7
- scikit-learn==0.24.2
- scipy==1.3.3
- seaborn==0.11.2
- tensorboard==2.10.1
- torch==1.10.2

# Usage: HPO

The Hyperparameter optimization (HPO) trial can be started by runing the “experiment.py” file. The change of configuration for this file is 

- If run locally:
    
    ```python
    hpo = input("Enter HPO Study type: ")    # GP, DGP or DSPP
    pruner_type = input("Enter Pruner Type: ")    # Optuna Pruner Type, either "None" or "HB" for Hyperband
    time_tolerance = int(input("Enter time tolerance: "))   # Maximum number of seconds before cancel a HPO trial
    ```
    
    ```
    Enter HPO Study type: DGP
    Enter Pruner Type: None
    Enter time tolerance: 6000
    ```
    

- If run on slurm: please change the configurations in the “experiment.py” file.

# Usage: file_read.py

The training data and test data extraction process can be done by the “file_read.py” file,  

# Usage: Model Training

The model training process can be started by runing the “train_GP.py”, “train_DGP.py” and “train_DSPP.py” file.

The trained model state file (*.pth file) and all the model parameters in *.txt file will be saved after training process is finished.

# Usage: Result Plot

The model result visulization can be shown by runing the “plot_GP_result_test.py”, “plot_DGP_result_test.py” and “plot_DSPP_result_test.py” file.
