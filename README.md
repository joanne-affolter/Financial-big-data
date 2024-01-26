# Market sentiment and in-homogeneous information flow : a cryptocurrency exchange case-study

***Joanne Affolter, João Luis Prado Vieira***
*School of Communication Systems, EPFL, Switzerland*

Welcome to our repository. The latter is organized as follows : 
- [**Abstract**](#Abstract)
- [**Repository Structure**](#repository-structure)
- [**Data**](#data)
- [**Reproducing our Results**](#reproducing-our-results)


## Abstract
Cryptocurrency markets are highly dynamic and volatile. They are characterized not only by rapid price fluctuations, but also by their sensitivity to market sentiment shifts relating jointly to the underlying assets and to the exchange platforms where these assets are traded. 

This study examines how market sentiment fluctuations influence changes in currency quotes across different cryptocurrency exchanges, focusing on Bitcoin. We collect tick-by-tick trade data and second-resolution market order book data from Binance and Bitstamp throughout Jan 2021 - Dec 2021 and employ Google Trends search indexes for crypto keywords as proxies. The analysis unfolds in three stages: (i) validating our chosen proxy as an explanatory variable for cryptocurrency price changes using transfer entropy as a measure of information flow, (ii) checking the validity of traditional stylized facts in the crypto case and their variations across both exchanges, and (iii) scrutinizing the time-evolution of the induced lead-lag correlation throughout the studied period.

---------------------------------------------------------
## Repository Structure

### Library 

These files contain all the functions used in the project, organized according to their respective main functionalities.

- **`library_data.py`** : Data collection and Data Pre-processing (Google Trends and Historical Trade data)
- **`library_coinapi_data_acquisition.py`** : Data collection and Data Pre-processing (Binance and Bitstamp data)
- **`library_plot.py`** : Generation of figures from the report. 
- **`library_stylized_facts.py`** : Financial stylized facts (logreturns, autocorrelation,...)
- **`library_stats.py`** : Basic statistics, statistical testing and estimation. 
- **`library_te.py`** : Entropy-based analyses (Bivariate Transfer Entropy, Multivariate Transfer Entropy)
- **`library_lead_lag_analysis.py`** : Lead-lag networks generation.
- **[library_pyRMT.py](https://github.com/GGiecold/pyRMT )** : Methods for correlation matrix cleaning with Random Matrix Theory. The code comes from  from Gregory Giecold's pyRMT library, which is currently unusable due to outdated PyPI configurations.
- **[/IDTxl](https://github.com/pwollstadt/IDTxl)** : Library used for efficient inference of networks and their node dynamics from multivariate time series data using information theory. 

### Notebooks
The code for execution is available in the following two notebooks, structured in alignment with the report. To ensure the pre-requirements are met before running the notebooks, please refer to the section [Reproducing our Results](#reproducing-our-results).

- **`Data_collection.ipynb`** : You will find in this notebook the code to be executed to collect data as reported in section II of the report.
- **`Results.ipynb`** : You will find in this notebook the code to be executed to obtain the results in section IV of the report.

### Report, figures and library requirements

- **`report.pdf`** : The final project report. 
- **`/figures`** : All figures used in the report. 
- **`requirements.txt`** : Library requirements. 

---------------------------------------------------------------
## Data 

The project's data is accessible on our Drive via the provided links. Note that the notebooks can be executed using the reduced dataset. We recommend doing so, as the full dataset is substantial in size.

- **Full data** : https://drive.google.com/drive/folders/1xTW6vvD-P_1pFwnT6bjecPkGabaqQ9Eb?usp=drive_link
- **Reduced data** : https://drive.google.com/drive/folders/10dJ2EkOrJz4WAlT8CxxwTPr3V7r1Z41Y?usp=drive_link

-------------------------------------------

## Reproducing our Results

1. Download this repository from Github (this is the .zip file you received by mail)
2. Go to the folder of the repository, open a terminal and run the following command 
    `pip install requirements.txt`
3. Download data from the Google Drive link provided above (Reduced data). 
4. Create a folder named `data` in the root folder and upload there the two directories from step 3 (`clean` and `raw`). The folder structure should look like this. 
```
    └── Financial-big-data
        ├── data
        ├── figures
        ├──  ...
        └── IDTxl
```
5. Run the notebook `Results.ipynb`

