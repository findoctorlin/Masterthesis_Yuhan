import optuna
import plotly.io as pio

def objective(trial):
    x = trial.suggest_float("x", -100, 100)
    y = trial.suggest_categorical("y", [-1, 0, 1])
    return x ** 2 + y


study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=10)

fig_history = optuna.visualization.plot_optimization_history(study)
fig_FI = optuna.visualization.plot_param_importances(study)
fig_slice = optuna.visualization.plot_slice(study)

pio.write_image(fig_history, '/home/linyuhan/Dokumente/Masterarbeit/dataset/CMAPSS/code/DGP_Yuhan/images/optuna_HPO_DGP_process/history.png')
pio.write_image(fig_FI, '/home/linyuhan/Dokumente/Masterarbeit/dataset/CMAPSS/code/DGP_Yuhan/images/optuna_HPO_DGP_process/FI.png')
pio.write_image(fig_slice, '/home/linyuhan/Dokumente/Masterarbeit/dataset/CMAPSS/code/DGP_Yuhan/images/optuna_HPO_DGP_process/slice.png')