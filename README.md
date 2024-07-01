# Bachelor Thesis 2024

Welcome to the repository for my Bachelor thesis in Econometrics and Economics. This is a brief summary of what can be found where.

## Installation

Prior to use, ensure to install the `thesis_tools` library. With pip, you may use `pip install -e .` after creating and activating a virtual environment.

## `thesis_tools` library
All relevant statistical models, as well as utility functions, are defined in the `thesis_tools` library. 

The `utils` directory features utility functions that can be used for data reading, as well as conversions of pandas dataframes to latex tables, and an easier way to fit models.

The `statistics` directory only contains a module to easily estimate hazard functions from data.

The `statistical_tests` directory contains modules that are used to easily compute a range of statistical tests, including added functionality such as the implementation of measurement errors.

The `models` directory defines all models, frequentist or Bayesian, used in the analysis.

## Data
All relevant data, alongside text files with links to the original sources, can be found in the `Data` directory.

## Figures
All figures used in the final paper can be found in the `Figures` directory.

## Notebooks
Any analysis for this thesis was performed in notebooks in the `Notebooks` directory.

Exploratory data analysis can be found in the `data` directory. Exploratory analysis of the hazard rates can be found in `explore/hazard_functions.ipynb`. The analysis of measurement errors is in the `extension` folder. 

Training and evaluation of all models is in the `models` directory, with self evident notebook names. Note that should one want to use pre-trained models, and stored intermediate results, the `../Stored_Models` and `../Stored_Results` folders are required. They are not uploaded to this repository due to the filesizes (~3.5GB), but will be made available upon request. 


