#! /bin/bash
# This script runs significance testing for protein tasks using the provided Python script.

#SBATCH --job-name=sig_test
#SBATCH --partition=superpod-a100
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=4:00:00
#SBATCH --output=/work/pi_annagreen_umass_edu/mahbuba/protein-tasks/sbatch_logs/%x_%j.out
#SBATCH --error=/work/pi_annagreen_umass_edu/mahbuba/protein-tasks/sbatch_logs/%x_%j.err


# Load necessary modules
module load conda/latest cuda/11.8
conda init

export TMPDIR=/tmp/$USER
mkdir -p $TMPDIR


conda activate /work/pi_annagreen_umass_edu/mahbuba/esmfold

python /work/pi_annagreen_umass_edu/mahbuba/protein-tasks/significance_testing_transformer.py