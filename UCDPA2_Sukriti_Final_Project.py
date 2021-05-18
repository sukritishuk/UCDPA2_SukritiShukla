
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


## Step 8 - Performing Machine Learning on life expectancy data using Different Models -
# Step 1 - Importing the necessary libraries to run the model and compute the Regression metrices for evaluation:
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

# Step 2 - Creating a Function to run different models by fitting each to training set and making predictions on testing set
# and visualizing the actual vs. predicted values as a plot:
# defining a Function model_fit_predict for all models:
def model_fit_predict(regressor):
    """Function to fit the Model to the training set and predict the Intercept, Coefficients and Regression Metrics for the
    Model on testing data, visualize a plot of actual vs. predicted values for each model"""

    # instantiating the regressor for the model:
    regress = regressor()

    # fitting the regressor to the training data using the fit method:
    regress.fit(life_feature_train, life_target_train)

    # predicting on the testing data using the predict method:
    life_pred = regress.predict(life_feature_test)

    # printing the Intercept generated from the model using the intercept_ attribute:
    print('Intercept:\n', regress.intercept_)

    # using matplotlib to plot Coefficients of different features determining Life Expectancy:
    plt.figure(figsize=(8, 5))

    # creating a line plot of coefficients generated by the model using the coef_ attribute:
    plt.plot(range(len(regress.coef_)), regress.coef_)

    # adding a horizontal red line across the x-axis:
    plt.axhline(0, color='r', linestyle='solid')

    # adding x-axis tick labels as all the feature column names:
    plt.xticks(range(len(life_feature.columns)), life_feature.columns, rotation=50)

    # adding a title to the plot:
    plt.suptitle("Coefficients from the Model")
    plt.title(regress.__class__.__name__)

    # displaying the plot
    plt.show()

    # Computing the key Regression Metrics (in sklearn) from testing data of Model and printing them:
    # printing the name of the Model used:
    print('Regression Metrics:', regress.__class__.__name__, '\n')
    # computing and printing the Root Mean Squared Error (RMSE) for the model:
    print('Root Mean Squared Error of Model: {:.2f}'.format(
        mean_squared_error(life_target_test, life_pred, squared=False)))
    # computing and printing the Mean Absolute Error (MAE) for the model:
    print('Mean Absolute Error of Model: {:.2f}'.format(mean_absolute_error(life_target_test, life_pred)))
    # computing and printing the Mean Squared Error (MSE) for the model:
    print('Mean Squared Error of Model: {:.2f}'.format(mean_squared_error(life_target_test, life_pred)))
    # computing and printing the R-squared value for the model:
    print('R_squared Score on actual vs. prediction: {:.2f}'.format(r2_score(life_target_test, life_pred)))

    # Visualizing the Actual vs. Predicted Life Expectancy values generated from the Model as a plot:
    # setting the figure and axes objects and figure sie for the plot:
    plt.figure(figsize=(14, 6))
    # adding a title to the plot by extracting a class name as a string to use as plot title for each Model:
    plt.suptitle('Visualizing the Actual vs. Predicted Life Expectancy values from the Model')
    plt.title(regress.__class__.__name__)
    # setting the x-axis range as length of actual target values in testing data:
    x_ax = range(len(life_target_test))
    # creating a scatter plot of actual target values:
    plt.scatter(x_ax, life_target_test, s=5, color="blue", label="actual")
    # creating a line plot for all the values target values predicted by the model:
    plt.plot(x_ax, life_pred, lw=0.8, color="red", label="predicted")
    # adding a legend to the plot:
    plt.legend()
    # displaying the plot:
    plt.show()

# A. Running Linear Regression Model on Life expectancy data:
# importing the model from the sklearn library:
from sklearn.linear_model import LinearRegression
# passing the first model as argument to the model_fit_predict function:
model_fit_predict(LinearRegression)    # calling the function with LinearRegression as argument

# B. Running Ridge Regression Model on Life expectancy data:
# importing the model from the sklearn library:
from sklearn.linear_model import Ridge
# passing the second model as argument to the model_fit_predict function:
model_fit_predict(Ridge)     # calling the function with Ridge as argument

# C. Running Lasso Regression Model on Life expectancy data:
# importing the model from the sklearn library:
from sklearn.linear_model import Lasso
# passing the second model as argument to the model_fit_predict function:
model_fit_predict(Lasso)     # calling the function with Lasso as argument

# D. Running Decision Tree Regression Model on Life expectancy data:
# importing the necessary libraries for the model:
from sklearn.tree import DecisionTreeRegressor
from sklearn import metrics

# We have already divided our Life Expectancy dataset into features and target labels.
# We have also split the data into training set (70%) and testing set (30%).

# instantiating a Decision tree regressor object, regress2:
regress2 = DecisionTreeRegressor(min_samples_leaf=0.12, random_state=42)  # each leaf containing at least 12% of the data used in training

# fitting the regressor with training data using fit method:
regress2.fit(life_feature_train, life_target_train)

# predicting on the testing set using the predict method:
decsn_tree_pred = regress2.predict(life_feature_test)

# computing the Regression Metrices (in sklearn) against testing data and printing the results:
print('Regression Metrics:', regress2.__class__.__name__, '\n')
print('Root Mean Squared Error:',
      np.sqrt(metrics.mean_squared_error(life_target_test, decsn_tree_pred)).round(decimals=2))
print('Mean Absolute Error: ', metrics.mean_absolute_error(life_target_test, decsn_tree_pred).round(decimals=2))
print('Mean Squared Error:', metrics.mean_squared_error(life_target_test, decsn_tree_pred).round(decimals=2))
print('R_squared Score on actual vs. prediction:',
      metrics.r2_score(life_target_test, decsn_tree_pred).round(decimals=2))

# storing each of the computed Regression Metrices in a variable
RMSE = mean_squared_error(life_target_test, decsn_tree_pred, squared=False).round(decimals=2)
MAE = mean_absolute_error(life_target_test, decsn_tree_pred).round(decimals=2)
MSE = mean_squared_error(life_target_test, decsn_tree_pred).round(decimals=2)
R2 = r2_score(life_target_test, decsn_tree_pred).round(decimals=2)

# importing the calculated metrics variables into a DataFrame for further use:
dec_tree_data = {'Model': [regress2.__class__.__name__], 'Root Mean Squared Error': [RMSE],
                 'Mean Absolute Error': [MAE],
                 'Mean Squared Error': [MSE], 'R-squared': [R2]}

dec_tree_df = pd.DataFrame(dec_tree_data, columns=['Model', 'Root Mean Squared Error', 'Mean Absolute Error',
                                                   'Mean Squared Error', 'R-squared'])



# E. Running Random Forest Regression Model on Life expectancy data:
# importing the necessary libraries for the model:
from sklearn.ensemble import RandomForestRegressor

# We have already divided our Life Expectancy dataset into features and target labels.
# We have also split the data into training set (70%) and testing set (30%).

# instantiating a Random Forest Regressor as 'rf' with following parameters:
rf = RandomForestRegressor(n_estimators=400, min_samples_leaf=0.12, random_state=42)
# n_estimators = 400 i.e.,number of trees in the forest as 400
# min_samples_leaf = 0.12 i.e., each leaf containing at least 12% of the data used in training

# fitting the regressor with training data using fit method:
rf.fit(life_feature_train, life_target_train)

# predicting on the testing set using the predict method:
rf_pred = rf.predict(life_feature_test)

# computing the Regression Metrices (in sklearn) against testing data and printing the results:
rmse_value = mean_squared_error(life_target_test, rf_pred,
                                squared=False)  # using sqaured parameter = False to compute square root

print('Regression Metrics:', rf.__class__.__name__, '\n')
print('Root Mean Squared Error: {:.2f}'.format(rmse_value))
print('Mean Absolute Error: {:.2f}'.format(mean_absolute_error(life_target_test, rf_pred)))
print('Mean Squared Error: {:.2f}'.format(mean_squared_error(life_target_test, rf_pred)))
print('R_squared Score on actual vs. prediction: {:.2f}'.format(r2_score(life_target_test, rf_pred)))

# storing each of the computed Regression Metrices in a variable:
RMSE = mean_squared_error(life_target_test, rf_pred, squared=False).round(decimals=2)
MAE = mean_absolute_error(life_target_test, rf_pred).round(decimals=2)
MSE = mean_squared_error(life_target_test, rf_pred).round(decimals=2)
R2 = r2_score(life_target_test, rf_pred).round(decimals=2)

rf_data = {'Model': [rf.__class__.__name__], 'Root Mean Squared Error': [RMSE], 'Mean Absolute Error': [MAE],
           'Mean Squared Error': [MSE], 'R-squared': [R2]}

# importing the calculated metrics variables into a DataFrame for further use:
rf_df = pd.DataFrame(rf_data, columns=['Model', 'Root Mean Squared Error', 'Mean Absolute Error',
                                       'Mean Squared Error', 'R-squared'])


# Visualizing the Actual vs. Predicted Life Expectancy values as Subplots from each of the Tree Models:
# setting up the matplotlib figure for subplots:
plt.figure(figsize=(18,6))
# adding a main title for all the subplots:
plt.suptitle('Visualizing the Actual vs. Predicted Life Expectancy values from the Tree Models')

# plotting the graph at the top for Decision Tree Regression Model:
plt.subplot(2, 1, 1)
# creating the plot for decision tree regression model as a scatter and line plot:
x_ax = range(len(life_target_test))
plt.scatter(x_ax, life_target_test, s=5, color="blue", label="actual")
plt.plot(x_ax, decsn_tree_pred, lw=0.8, color="red", label="predicted")
# adding a title to the subplot:
plt.title(regress2.__class__.__name__)   # extracting a class name as a string to use as plot title
# adding a legend to the plot:
plt.legend()

# # plotting the graph at the top for Random Forest Regression Model:
plt.subplot(2, 1, 2)
# creating the plot for random forest regression model as a scatter and line plot:
x_ax = range(len(life_target_test))
plt.scatter(x_ax, life_target_test, s=5, color="blue", label="actual")
plt.plot(x_ax, rf_pred, lw=0.8, color="green", label="predicted")
# adding a title to the subplot:
plt.title(rf.__class__.__name__)   # extracting a class name as a string to use as plot title
# adding a legend to the plot:
plt.legend()
# displaying the plot:
plt.show()