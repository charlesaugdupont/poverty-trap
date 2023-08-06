#!/bin/bash

# Set job requirements
#SBATCH --nodes=1
#SBATCH --ntasks=10
#SBATCH --partition=thin
#SBATCH --time=01:30:00

# Load modules
module load 2021
module load Python/3.9.5-GCCcore-10.3.0
pip install --user scipy

# Create output directory on scratch
mkdir "$TMPDIR"/concat_G_arrays_cpt_sda_fixed_w_scale &

# Start jobs
for i in `seq 1 $SLURM_NTASKS`; do
	srun --ntasks=1 --nodes=1 --cpus-per-task=1 --mem-per-cpu=4G python -W ignore concat_G_arrays_cpt_sda_fixed_w_scale.py "$TMPDIR"/concat_G_arrays_cpt_sda_fixed_w_scale $i &
done
wait

# Copy output data to home directory
cp -r "$TMPDIR"/concat_G_arrays_cpt_sda_fixed_w_scale $HOME