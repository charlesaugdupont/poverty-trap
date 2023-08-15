#!/bin/bash

# Set job requirements
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=thin
#SBATCH --time=02:00:00

# Load modules
module load 2021
module load Python/3.9.5-GCCcore-10.3.0
pip install --user scipy

# Create output directory on scratch
mkdir "$TMPDIR"/concat_W_arrays_valya &

# Start jobs
srun --ntasks=1 --nodes=1 --cpus-per-task=1 --mem-per-cpu=4G python -W ignore concat_W_arrays_valya.py "$TMPDIR"/concat_W_arrays_valya &
wait

# Copy output data to home directory
cp -r "$TMPDIR"/concat_W_arrays_valya $HOME