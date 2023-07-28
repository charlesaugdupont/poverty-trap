#!/bin/bash

# Set job requirements
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=thin
#SBATCH --time=04:00:00

# Load modules
module load 2021
module load Python/3.9.5-GCCcore-10.3.0
pip install --user scipy

# Create output directory on scratch
mkdir "$TMPDIR"/concat_W_arrays &

# Start jobs
for i in `seq 1 $SLURM_NTASKS`; do
	srun --ntasks=1 --nodes=1 --cpus-per-task=1 --mem-per-cpu=32G python -W ignore concat_W_arrays.py "$TMPDIR"/concat_W_arrays &
done
wait

# Copy output data to home directory
cp -r "$TMPDIR"/concat_W_arrays $HOME