# Reproducibility code for "Political attitudes differ but share a common low-dimensional structure across social media and survey data"

To reproduce the analysis from the paper, follow the steps detailed below.

Tested with: 
- Python 12.2
- R 4.5.0

`functions.py` contains custom Python functions used in `analysis.ipynb`.

The figures created by the Python programs are included in the `plots` folder.

## 1. Data acquisition
First, you need to download the X data and the ESS data and place it in the `data` folder.

### 1.1. Get X data
To download the X data, proceed as follows.
1. Go to the [OSF repository](https://doi.org/10.17605/OSF.IO/AT5Q2).
2. Select and download the three files: `followers_positions.csv`, `followers_activity.csv`, `followers_impression_counts.csv`.
3. Move the files to the `data` folder.

### 1.2. Get European Social Survey (ESS) data
To download the results of the 11th wave of the ESS, proceed as follows.
1. Go to the [ESS 11 data portal](https://doi.org/10.21338/ess11e04_1).
2. Click the Download button.
3. You will be asked to register on the ESS data portal. This is free.
4. Select `.csv` format to download.
5. Unzip and move the `.csv` file to the `data` folder.

> Our analysis was performed using the edition 4.1 of the data, released on January 13, 2026.

### 1.3. Pre-process ESS data
- Run `ESS_preprocessing.ipynb`.

This removes non-France countries, missing values, it renames variables, etc. This creates two new files in the `data` folder: `ESS11_preprocessed.csv`, and `ESS11-pao_preprocessed.csv`. The second one is not necessary for running the main analysis, it is only used in the Supplementary Material.

## 2. Analysis
For the main analysis, we first perform computations for the ESS data in R, to take into account survey weighting. Then we compare with the X data in Python.

### 2.1. Perform computations for ESS data

- Run `ESS_computations.R`.

Because we need to take survey weighting into account, we run computations pertaining to the ESS dataset in R. Results are exported to `.csv` files for later analysis in Python. 

>This step is optional, as the `.csv` files are already provided in the repository.

### 2.1. Perform computations for X data and compare with ESS

- Run `analysis_main.ipynb`.

Finally, once we have the results from the ESS computations, we can analyze them together with the X dataset.


## 3. Supplementary
For the supplementary, we first need to perform computations for the ESS-pao subpanel in R. Then we analyze the results in Python.

### 3.1. Perform computations for ESS-pao subpanel

- Run `ESS-pao_computations.R`.

Because we need to take survey weighting into account, we run computations pertaining to the ESS-pao subpanel in R. Results are exported to `.csv` files in `ESS_results/pao_panel/` for later analysis in Python.

>This step is optional, as the `.csv` files are already provided in the repository.


### 3.2. Perform supplementary analyses

- Run `analysis_supplementary.ipynb`.
