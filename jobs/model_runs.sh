#!/bin/bash

# Set job requirements
#SBATCH --nodes=1
#SBATCH --ntasks=128
#SBATCH --partition=thin
#SBATCH --time=04:30:00

# Load modules
module load 2021
module load Python/3.9.5-GCCcore-10.3.0
pip install --user scipy
pip install --user SALib
pip install --user networkx
pip install --user pymarkowitz

# Create output directory on scratch
mkdir "$TMPDIR"/model_runs &

# Start jobs
for i in `seq 1 $SLURM_NTASKS`; do
	srun --ntasks=1 --nodes=1 --cpus-per-task=1 python -W ignore network_simulation.py "$TMPDIR"/model_runs $i &
done
wait

# Copy output data to home directory
cp -r "$TMPDIR"/model_runs $HOME