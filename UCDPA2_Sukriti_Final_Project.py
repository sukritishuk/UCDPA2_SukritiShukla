
### Data Source 1 - Data from Kaggle Public Dataset: Life Expectancy (WHO)

## Step 1 - Importing the necessary libraries -
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

## Step 2 - Loading the dataset from Kaggle and getting summary information -
# Loading the data from csv file into Pandas and reading it using Pandas library (alias as pd):
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

## Step 3 - Cleaning and Formatting the data -
# Removing spaces at both ends in the column names:
life_data.columns = life_data.columns.str.strip()
print(life_data.columns)
# Renaming the columns and reformatting them:
life_data.rename(columns = {'Life expectancy' : 'Life Expectancy','infant deaths': 'Infant deaths','under-five deaths':'Under-five deaths','thinness  1-19 years':'Thinness 1-19 years','thinness 5-9 years':'Thinness 5-9 years'},inplace=True)
# Checking for any Missing Values (NaN) in each column and getting the total of such missing rows for each column:
print(life_data.isnull().sum())
# Imputing missing values with the mean value of the column:
life_data.fillna(value = life_data.mean(), inplace = True)
# Re-calculating the basic summary statistics of dataset post-imputation of missing values:
print(life_data.describe())
# Re-checking for sum of Missing values in each column after imputation:
print(life_data.isnull().sum())


## Step 4 - Converting Categorical data to numeric values using One-Hot Encoding -
# Encoding Dummy variables for 'Status' column: using get_dummies function:
life_data_status = pd.get_dummies(life_data.Status)
# Printing the columns of life_data_status:
print(life_data_status.columns)
# Concatenating the data after encoding dummy variables:
life_data = pd.concat([life_data, life_data_status], axis = 1)
# Renaming the Dummy variable columns created after one-hot encoding:
life_data.rename(columns = {'Developed' : 0, 'Developing' : 1})
# Getting a snapshot of first 3 rows of the data using the head() method, after One-hot encoding is applied::
print(life_data.head(3))

# Chart 1 - Visualizing Life Expectancy in Developed vs. Developing countries as a Violin plot:
# creating the violin plot using seaborn (alias as sns) with Status dummy variables on x-axis, Life expectancy on y-axis
sns.violinplot(x="Status", y="Life Expectancy", data=life_data,palette='rainbow')
# adding a title to the plot:
plt.title('Life Expectancy in Developing vs. Developed countries')
# displaying the plot:
plt.show()


## Step 5 - Grouping the Life Expectancy data and Understanding the Relation among Factors -
# Grouping Type 1 - Grouping Data by Status:
# removing the Year column from the data using drop method:
life_data_comb = life_data.drop('Year', axis = 1)
# grouping the data by Status and aggregating by mean of each factor column, using groupby and mean functions:
life_data_status = life_data_comb.groupby(['Status']).mean().round(decimals=2)  # rounding values to 2 decimal places
# Getting a snapshot of first 3 rows of the grouped data using the head() method:
print(life_data_status.head(3))

# Grouping Type 2 - Grouping Data by Countries:
# grouping the data by country and aggregating by mean of each factor column, using groupby and mean functions:
life_data_country = life_data_comb.groupby('Country').mean().round(decimals=2)  # rounding values to 2 decimal places
# Getting a snapshot of first 5 rows of the grouped data using the head() method:
print(life_data_country.head())

# Chart 2 - Understanding the Correlation among different factors as a Heatmap:
# creating a Figure and Axes object for the plot and specifying the size of the figure:
plt.figure(figsize = (15,6))
# creating a heatmap of different factors, using the heatmap function and corr() method:
sns.heatmap(life_data_comb.corr(), annot = True,annot_kws={"size":8})
# adding a title to the plot:
plt.title('Correlation among different Life Expectancy-related Factors')
# displaying the plot:
plt.show()


