#!/bin/bash

# Set job requirements

#SBATCH -p thin
#SBATCH -t 01:00:00

#SBATCH --nodes=1
#SBATCH --ntasks=16

#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=c.a.dupont@uva.nl

# Load modules
module load 2021
module load Python/3.9.5-GCCcore-10.3.0
pip install --user scipy
pip install --user SALib
pip install --user networkx
pip install --user git+https://github.com/cvxgrp/cptopt.git
pip install --user pymarkowitz


# Create output directory on scratch
mkdir "$TMPDIR"/output_directory &

for i in `seq 0 15`; do
	# Execute python script
	python network_simulation.py "$TMPDIR"/output_directory $i &
done
wait

# Copy output data to home directory
cp -r "$TMPDIR"/output_directory $HOME