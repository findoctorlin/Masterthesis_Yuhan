#!/bin/bash

#SBATCH --mail-user=linyuhan@tnt.uni-hannover.de     # only<UserName>@tnt.uni-hannover.de is allowed as mail address
#SBATCH --mail-type=ALL                              # Eine Mail wird bei Job-Start/Ende versendet
#SBATCH --job-name=FD001:hpo:optuna:v2               # Name unter dem der Job in der Job-History gespeichert wird
#SBATCH --output=/home/linyuhan/Dokumente/Masterarbeit/dataset/CMAPSS/code/DGP_Yuhan/slurm_log/slurm-%j-out.log 
                                                     # Logdatei für den merged STDOUT/STDERR output (%j wird durch slurjob-ID ersetzt)

#SBATCH --time=24:00:00                              # Maximale Laufzeit des Jobs, bis Slurm diesen abbricht (HH:MM:SS)
#SBATCH --partition=cpu_normal_stud               # Partition auf der gerechnet werden soll. Ohne Angabe des Parameters wird auf der
                                                     # Default-Partition gerechnet. Es können mehrere angegeben werden, mit Komma getrennt.
#SBATCH --nodes=1                  # Reservierung von 1 Rechenknoten, alle nachfolgend reservierten 
                                   # CPUs müssen sich auf den reservierten Knoten befinden
#SBATCH --tasks-per-node=1        # Reservierung von 16 CPUs pro Rechenknoten
#SBATCH --mem=16G                  # Reservierung von 32GB RAM

trap "kill 0" SIGINT

working_dir=/home/linyuhan/Dokumente/Masterarbeit/dataset/CMAPSS/code/DGP_Yuhan/; cd $working_dir
env_name=CMAPSS
a_python=/usr/bin/python3.10

echo "The job has begun"
# export PYTHONPATH=$BIGWORK/workspace/UIDA:$PYTHONPATH
# echo PYTHONPATH=$PYTHONPATH

file_to_run='/home/linyuhan/Dokumente/Masterarbeit/dataset/CMAPSS/code/DGP_Yuhan/experiment.py'

srun $a_python $file_to_run --hpo DGP --file CMAPSS --pruner_type HB --time_tolerance 3600

#########################
# save_dir_prefix='/home/linyuhan/Dokumente/Masterarbeit/dataset/CMAPSS/code/DGP_Yuhan/'

# $a_python $file_to_run --model DACNN --data CMAPSS-FD001:CMAPSS-FD002 
# --base_config_file experiment_DANN_CMAPSS_v1.2.ini --save_dir_prefix 
# $save_dir_prefix --grl_lambda auto --hparams_opt learn_rate,weight_decay 
# --hpo_decision_metric v_so_task_rmse,v_domain_conf_loss 
# --hpo_decision_direction minimize,minimize --n_trials 100 
# --criterion_task L1Loss  --criterion_domain BCELoss --max_epoch 100 
# --seed_num 0 --output_transform "normalize"

########################
