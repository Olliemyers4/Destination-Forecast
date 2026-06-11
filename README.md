# Destination-Forecast

## Setup
This code has been developed and tested using python 3.11.

Required Packages can be installed via:
`pip install requirements.txt`

If developing on this repo, also install the development packages via:
`pip install requirements-dev.txt`

**To be able to run this code you will a source of the required datasets. They must be placed into the root directory of this repository**

## Studying the dataset

The notebook `dataset_exploration.ipynb` is used to explore the datasets provided.

The aim of the notebook is to demonstrate the process taken to understand the datasets thoroughly
The below VS Code extension alongside the notebooks allows for a more interactive view of tables produced by code cells: 
[VS Code Extension](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.datawrangler)

Investigation into dataset revealed a number of patterns that could be exploited for forecasting destinations

## Destination Forecast Model

The segment of this repository under `/src/` aims to create an early prototype of a model to predict the next destination of a vessel.

Files:

- `src/forecaster.py` All the classes and methods needed for executing training

- `src/training.py` Instantiation of classes and calling of the training

- `src/dataset_preparation/dataset_creator.py` Generates train test validation splits

- `src/dataset_preparation/dataset_data_label_split.pt` Utility function to split data files into data and labels



