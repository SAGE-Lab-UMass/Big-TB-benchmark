#!/bin/bash
export TF_ENABLE_ONEDNN_OPTS=0

# Assign the gene and group
mdcnn_file="model_training/main_mdcnn_crossval.py"
param_file="model_training/parameter_files/params_filter12_epoch250.txt"

# Python command to run the run_mdcnn script
python $mdcnn_file $param_file