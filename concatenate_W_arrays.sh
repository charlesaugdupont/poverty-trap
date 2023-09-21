#!/bin/bash

# Set job requirements
#SBATCH --nodes=1
#SBATCH --ntasks=20
#SBATCH --partition=thin
#SBATCH --time=01:30:00
#SBATCH --mem=160GB


# Load modules
module load 2021
module load Python/3.9.5-GCCcore-10.3.0
pip install --user scipy

# Create output directory on scratch
mkdir "$TMPDIR"/W_arrays &

# Start jobs
for i in `seq 1 $SLURM_NTASKS`; do
	srun --ntasks=1 --nodes=1 --cpus-per-task=1 --mem-per-cpu=8G python -W ignore concatenate_W_arrays.py "$TMPDIR"/W_arrays $i &
done
wait

# Copy output data to home directory
cp -r "$TMPDIR"/W_arrays $HOME