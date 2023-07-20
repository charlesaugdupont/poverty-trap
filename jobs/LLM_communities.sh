#!/bin/bash

# Set job requirements
#SBATCH --nodes=1
#SBATCH --ntasks=128
#SBATCH --partition=thin
#SBATCH --time=06:00:00

# Load modules
module load 2021
module load Python/3.9.5-GCCcore-10.3.0
pip install --user scipy
pip install --user gldpy

# Create output directory on scratch
mkdir "$TMPDIR"/LLM_communities &

# Start jobs
for i in `seq 1 $SLURM_NTASKS`; do
	srun --ntasks=1 --nodes=1 --cpus-per-task=1 --mem-per-cpu=4G python -W ignore llm_communities.py "$TMPDIR"/LLM_communities $i &
done
wait

# Copy output data to home directory
cp -r "$TMPDIR"/LLM_communities $HOME