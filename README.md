# NMRlipids Databank Usage Templates

Separated repository collecting usage examples of [NMRlipids Databank](https://github.com/NMRlipids/Databank) API.
To consult the API documentation, please visit https://nmrlipids.github.io/databankLibrary.html.

Before using these templates locally, you need to clone the NMRlipids databank https://github.com/NMRLipids/Databank and provide its location in the first cell.

Notebooks can be also ran in Colab without any installation.

## template.ipynb

Notebook demonstrate the basic usage examples: 
- loading the Databank
- viewing metadata for a simulation
- viewing precomputed data for the simulation
- download the trajectory and computes P-N angle from it.

## plotQuality.ipynb

Notebook shows ranking tables of simulations based on quality evaluation against experiments.


## Form factor to total density notebook 
The notebook shows how a neural net can be trained to predict total electron densities from SAXS form factors using data in the NMRlipids databank.

The notebook contains the following steps:
1) Download the data using the [NMR lipids API](https://nmrlipids.github.io/databankLibrary.html).  
2) Explore the data, and preprocess the data for the machine learning pipeline.  
3) Split data into train and test sets.
4) Implement different neural networks and perform hyperparameter tuning
5) Train and evaluate the performance of the neural networks. 

### Installing dependencies
The form factor to total density notebook uses Poetry for package handling. Dependencies for running the notebook can be seen in pyprojects.toml, and can be installed using:
\```bash
poetry install
\```

### Background 
The notebook is based on initial work by the [CellScatter project](https://github.com/K123AsJ0k1/CellScatter/tree/main). 

## library.py

Example of how complicated things can be extracted from the Databank (under construction)..
