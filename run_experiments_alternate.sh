#!/bin/bash

# Set job requirements
#SBATCH --nodes=1
#SBATCH --ntasks=128
#SBATCH --partition=genoa
#SBATCH --time=10:00:00

# Load modules
module load 2022
module load Python/3.10.4-GCCcore-11.3.0
pip install --user scipy
pip install --user SALib
pip install --user cvxpy
pip install --user networkx

args=("$@")
 
# Create output directory on scratch
mkdir "$TMPDIR"/model_runs_alternate_${args[0]} &

# Start jobs
for i in `seq 1 $SLURM_NTASKS`; do
	srun --ntasks=1 --nodes=1 --cpus-per-task=1 python -W ignore simulation_alternate.py "$TMPDIR"/model_runs_alternate_${args[0]} $i ${args[0]} &
done
wait

# Copy output data to home directory
cp -r "$TMPDIR"/model_runs_alternate_${args[0]} $HOME
