#!/bin/bash
export TF_ENABLE_ONEDNN_OPTS=0

# Assign the gene and group
sdcnn_file="model_training/run_SDCNN_ccp_crossval.py"
# param_file="model_training/parameter_files/optimized_epochs/MOXI_ccp_epoch_60.txt"
param_file="model_training/parameter_files/optimized_epochs/CAP_ccp_epoch_120.txt"


# Python command to run the run_mdcnn script
python $sdcnn_file $param_file
