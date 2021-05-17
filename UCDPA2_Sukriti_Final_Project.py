
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


# Computing the correlation of each factor with Life Expectancy -
corr_target = life_data_comb.corr()["Life Expectancy"]
# selecting the most highly correlated features or variables from the data, both Positive and Negative:
features_corr_positive = corr_target[corr_target > 0.5]
features_corr_negative = corr_target[corr_target <= -0.5]
# printing the most postively and most negatively correlated factors:
print('Most Positively Correlated Features:''\n',features_corr_positive[features_corr_positive.values != 1.0].round(decimals=2))
print('Most Negatively Correlated Features:''\n',features_corr_negative.round(decimals=2))



# Chart 3 - Visualizing the relation between Life Expectancy and different factors as Scatter Subplots:
# creating a Figure and Axes object for the plot and specifying the size of the figure:
f, axes = plt.subplots(3, 3, figsize=(20,15))
# creating each scatter subplot color encoded by Status (i.e., Developed vs. Developing) using Seaborn:
sns.scatterplot(data=life_data_comb, x="Adult Mortality",y='Life Expectancy',hue='Status',ax=axes[0, 0])
sns.scatterplot(data=life_data_comb, x="Infant deaths",y='Life Expectancy',hue='Status',ax=axes[0, 1])
sns.scatterplot(data=life_data_comb, x="Alcohol",y='Life Expectancy',hue='Status',ax=axes[0, 2])
sns.scatterplot(data=life_data_comb, x="Hepatitis B",y='Life Expectancy',hue='Status',ax=axes[1, 0])
sns.scatterplot(data=life_data_comb, x="BMI",y='Life Expectancy',hue='Status',ax=axes[1, 1])
sns.scatterplot(data=life_data_comb, x="Schooling",y='Life Expectancy',hue='Status',ax=axes[1, 2])
sns.scatterplot(data=life_data_comb, x="GDP",y='Life Expectancy',hue='Status',ax=axes[2, 0])
sns.scatterplot(data=life_data_comb, x="Population",y='Life Expectancy',hue='Status',ax=axes[2, 1])
sns.scatterplot(data=life_data_comb, x="Income composition of resources",y='Life Expectancy',hue='Status',ax=axes[2, 2])
# adding a main title for all the subplots:
plt.suptitle('Relation among different Factors and Life Expectancy (Developed vs. Developing countries)',fontsize=14)
# adjusting the spacing between main title and subplots:
plt.subplots_adjust(top=0.95,hspace=0.4,wspace=0.2)
# displaying the plot:
plt.show()


# Chart 4 - Life Expectancy trends for Developed vs. Developing countries as a Boxplot:
#  creating a Figure and Axes object for the plot and specifying the size of the figure:
plt.figure(figsize = (20, 15))
# creating a boxplot for the period (2000-15) using Seaborn boxplot:
sns.boxplot(x="Year", hue="Status", y="Life Expectancy", data=life_data)
# adding a title to the plot:
plt.title('Life Expectancy trends for Developed vs. Developing countries', fontsize=14)
# displaying the plot:
plt.show()



## Step 6 - Understanding the distribution of Life Expectancy using empirical distributions -
# importing Allen Downey's python library representing empirical distributions:
from empiricaldist import Cdf
# creating a Figure and Axes object for the plot and specifying the size of the figure:
plt.figure(figsize=(15,8))
# slicing the data for Developed and Developing nations:
Developing = life_data_comb['Status'] == 'Developing'
# slicing life expectancy data for each of the developed & developing nation groups and storing each in a variable:
Life_Expect = life_data_comb['Life Expectancy']
developing_data = Life_Expect[Developing]
developed_data =  Life_Expect[~Developing]
# plotting the Cumulative Distribution Function (CDFs) of Life Expectancy as line plot using Cdf function:
Cdf.from_seq(developing_data).plot(label='Developing')
Cdf.from_seq(developed_data).plot(label='Developed')
# adding x-axis and y-axis labels:
plt.xlabel('Life Expectancy (years)')
plt.ylabel('CDFs')
# adding a title to the plot:
plt.title('Cumulative Distribution of Life Expectancy for Developed vs. Developing countries',fontsize=14)
# displaying the plot:
plt.show()


## Step 7 - Preparing the data for Machine Learning -
# Step i) Grouping Life Expectancy data by Country and aggregating to show the mean between 2000-15 for each Factor -
# dropping the Status & Year column, grouping by Country and aggregating each Factor by the mean of all Years:
processed_life_data = life_data.drop(['Status','Year'], axis = 1).groupby(['Country']).mean().round(decimals=2)
# studying the data size (total rows & columns) using the shape attribute after cleaning it:
print(processed_life_data.shape)


# Step ii) Segregating the data into Feature and Target variables -
# slicing the Life Expectancy column as the Target variable and storing it in a variable, life_target:
life_target = processed_life_data['Life Expectancy']
# getting a snapshot of first 5 rows of the sliced Pandas Series for the Target variable using the head method:
print(life_target.head())
# slicing all the remaining columns as Feature variables and storing them in another variable, life_feature:
life_feature = processed_life_data.drop(['Life Expectancy'], axis = 1)
# getting a snapshot of first 5 rows for all the Feature variables:
print(life_feature.head())
# printing the total size of the Target Variables and the Feature Variables using the shape attribute:
print('Shape of Target Variable:',life_target.shape, 'Shape of Feature Variables :',life_feature.shape)


# Step iii) Feature Scaling to align the Varying Data Ranges using MinMaxScaler() -
# importing the necessary package from scikit-learn library to scale the features:
from sklearn.preprocessing import MinMaxScaler
# instantiating the scaling object and assigning it to a variable:
scaler = MinMaxScaler()
# fitting the scaling object to Features Variable DataFrame and transforming the data using the fit_transform method:
scaled_life_feature = scaler.fit_transform(life_feature)
# returning the scaled and transformed Features as a Numpy array:
print(scaled_life_feature)


# Step iv) Splitting Preprocessed Machine Learning data into Training & Testing Sets -
# Using Scaled featured and a split ratio of 70:30 -
# splitting the Scaled Features and Target variable sets into training dnd testing data -
# importing the necessary libraries to perform the split:
from sklearn.model_selection import train_test_split
# creating the training and test sets using the train_test_split function and by specifying split ratio 70-30:
life_feature_train, life_feature_test, life_target_train, life_target_test = train_test_split(
scaled_life_feature, life_target, test_size=0.3,random_state=42)  # using the parameter random_state for reproducibility
# printing the length of train and test variables created after the split using the Python's built-in len function:
print('X_train:',len(life_feature_train),'y_train:',len(life_target_train))
print('X_test:',len(life_feature_test), 'y_test:',len(life_target_test))
