#!/bin/bash

# Define the parameter file
parameter_file="interpretability/parameter_files/logreg_iters1000.txt"

# Run the Python script, passing the config file as an argument
python interpretability/run_interpret_logreg.py $parameter_file