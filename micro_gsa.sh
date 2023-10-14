#!/bin/bash

# Set job requirements
#SBATCH --nodes=1
#SBATCH --ntasks=20
#SBATCH --partition=thin
#SBATCH --time=6:00:00

# Load modules
module load 2021
module load Python/3.9.5-GCCcore-10.3.0
pip install --user scipy
pip install --user UQpy==4.0.5
pip install --user scikit-learn

# Create output directory on scratch
mkdir "$TMPDIR"/micro_gsa &

# Start jobs
for i in `seq 1 $SLURM_NTASKS`; do
	srun --ntasks=1 --nodes=1 --cpus-per-task=1 python -W ignore micro_gsa.py "$TMPDIR"/micro_gsa $i &
done
wait

# Copy output data to home directory
cp -r "$TMPDIR"/micro_gsa $HOME