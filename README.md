# Setup

- Clone the repository 
    - `git clone git@github.com:charlesaugdupont/poverty-trap.git`
- Create a virtual environment
    - `python -m venv venv`
- Activate it
    - `source venv/bin/activate`
- Install dependencies
    - `pip install -r requirements.txt`

# Experiments
- Generate Social Distance Attachment (SDA) graphs
    - `python generate_sda_graphs.py --graph_dir graphs`
- Run the model
    - `python simulation.py --output_dir output --chunk_idx 1 --seed_idx 0 --graph_dir graphs`
    - Note: the above only runs a small portion of all parameter combinations, and for only one random seed. In order to generate 
all the necessary data, care must be taken to run all 128 chunks, and each one for 20 different random seeds.


