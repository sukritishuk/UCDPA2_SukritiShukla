
# Project Scope - Predicting Life Expectancy using Machine Learning Algorithms in Python -
### Data Source 1 - Data from Kaggle Public Dataset: Life Expectancy (WHO)

## Step 1 - Importing the necessary libraries -
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

## Step 2 - Loading the dataset from Kaggle and getting summary information -
# Loading the data from csv file into Pandas and reading it:
life_data = pd.read_csv('Life Expectancy Data.csv')
# Getting a snapshot of first 5 rows of the data using the head() method:
print(life_data.head())
# Studying the data size (total rows & columns) using the shape attribute:
print(life_data.shape)
# Summarizing the basic information about the data using info() method:
print(life_data.info())
# Displaying the column headers to check if the column names are in correct format:
print(life_data.columns)
# Getting a summary about the data with all the basic summary statistics using the describe() method:
print(life_data.describe())

