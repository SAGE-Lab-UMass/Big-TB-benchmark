#!/bin/bash

# Define the parameter file
parameter_file="model_training/parameter_files/logreg_iters1000.txt"

# Run the Python script, passing the config file as an argument
python model_training/run_logreg_l2.py $parameter_file
