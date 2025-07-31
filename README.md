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
    - Running all simulations will yield a set of 20 different folders, named *seed_0*, *seed_1*, $\ldots$, *$seed\_19$*
    - These resulting raw data may be found at https://figshare.com/s/23efbca44d7b28ac0340

# Figures and Analysis

### Preliminary Data Processing
In order to carry out analysis and replicate figures from the article, the following notebooks must be run to process raw data.
- `notebooks/ind_degree_final_wealth.ipynb`
    - Generates `regime.pickle`, `agent_comm_degrees.pickle`, `agent_final_wealths.pickle`
- `notebooks/attention_results.pickle`
    - Generates `attention_results.pickle`, `attention_results_by_comm.pickle`

### Figure Replication
The following list states where relevant code can be found to generate key figures from the article.
- Figure 1: `notebooks/Network Figures.ipynb`
- Figure 2: (not generated with code)
- Figure 3: `notebooks/CPT.ipynb`
- Figure 4: `notebooks/radar_plots.ipynb`
- Figure 5: `notebooks/ind_degree_final_wealth.ipynb`
- Figure 6: `notebooks/ind_degree_final_wealth.ipynb`
- Figure 7: `notebooks/radar_plots_community.ipynb`
- Figure 8: `notebooks/attention_results.ipynb`
- Figure 9: `notebooks/Intervention.ipynb`
- Supplementary Information Figure 1: `notebooks/GSA_Results.ipynb` 
- Supplementary Information Figure 2: `notebooks/GSA_Results.ipynb` 
- Supplementary Information Figure 3: `notebooks/sda_graph_distributions.ipynb` 
- Supplementary Information Figure 4: `notebooks/Paper_Project_Returns.ipynb` 
- Supplementary Information Figure 5: `notebooks/ind_degree_final_wealth.ipynb` 
- Supplementary Information Figure 6: `notebooks/Sum_of_bimodals.ipynb` 
- Supplementary Information Figure 7: `notebooks/comm_degree_median_wealth.ipynb`
- Supplementary Information Figure 8: `notebooks/CCDF.ipynb`
