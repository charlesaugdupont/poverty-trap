# Emergent poverty traps at multiple levels impede social mobility
The code inside of this repository accompanies the paper with the above title.

## Setup
- Clone the repository 
    - `git clone git@github.com:charlesaugdupont/poverty-trap.git`
- Create a virtual environment
    - `python -m venv venv`
- Activate it
    - `source venv/bin/activate`
- Install dependencies
    - `pip install -r requirements.txt`

## Experiments
- Navigate to `src` directory
- Generate Social Distance Attachment (SDA) graphs
    - `python generate_sda_graphs.py --graph_dir graphs`
- Run the model
    - `python simulation.py --output_dir output --chunk_idx 1 --seed_idx 0 --graph_dir graphs`
    - Note: the above only runs a small portion of all parameter combinations, and for only one random seed. In order to generate 
all the necessary data, care must be taken to run all 128 chunks, and each one for 20 different random seeds.

# Figures and Analysis
The following list states where relevant code can be found to generate key figures from the article.
- Figure 2: `notebooks/radar_plots.ipynb` 
- Figure 3: `notebooks/ind_degree_final_wealth.ipynb` 
- Figure 4: `notebooks/ind_degree_final_wealth.ipynb` 
- Figure 5: `notebooks/attention_results.ipynb` 
- Figure 6: `notebooks/Intervention.ipynb` 
- Figure 7: `notebooks/radar_plots_community.ipynb` 
- Figure 8a: `notebooks/Network Figures.ipynb` 
- Figure 10: `notebooks/CPT.ipynb` 
- Figure 11: `notebooks/GSA_Results.ipynb` 
- Figure 12: `notebooks/GSA_Results.ipynb` 
- Figure 13: `notebooks/sda_graph_distributions.ipynb` 
- Figure 14: `notebooks/Paper_Project_Returns.ipynb` 
- Figure 15: `notebooks/ind_degree_final_wealth.ipynb` 
- Figure 16: `notebooks/Sum_of_bimodals.ipynb` 
- Figure 17: `notebooks/comm_degree_median_wealth.ipynb`
