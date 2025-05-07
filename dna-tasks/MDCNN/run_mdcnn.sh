#!/bin/bash
export TF_ENABLE_ONEDNN_OPTS=0

# Assign the gene and group
mdcnn_file="model_training/main_mdcnn_crossval.py"
param_file="model_training/parameter_files/params_filter12_epoch250.txt"

# conda init
# conda activate /work/pi_annagreen_umass_edu/saishradha/miniconda3/envs/cnn
# module load cuda/11.8.2 cudnn/8.7.0.84-11.8 

# Python command to run the run_mdcnn script
python $mdcnn_file $param_file