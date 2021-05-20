
### Data Source 1 - Data from Kaggle Public Dataset: Life Expectancy (WHO)

## Step 1 - Importing all the basic Python libraries for Data Analysis & Visualization -
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
# Step i) - Importing the necessary libraries to run the model and compute the key Regression metrices for Model evaluation:
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_percentage_error

# Step ii) - Creating a Function to run different models by fitting each and making predictions on testing set;; computing key metrices
# visualizing the actual vs. predicted values as a plot:
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
    # computing and printing the Mean Absolute Percentage Error (MAPE) for the model:
    print('Mean Absolute Percentage Error of Model: {:.2f}'.format(mean_absolute_percentage_error(life_target_test, life_pred)))

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

# Step iii) Calling the model_fit_predict function with each model name as argument -
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
# passing the third model as argument to the model_fit_predict function:
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
print('Mean Absolute Percentage Error: : {:.2f}'.format(mean_absolute_percentage_error(life_target_test, decsn_tree_pred)))

# storing each of the computed Regression Metrices in a variable
RMSE = mean_squared_error(life_target_test, decsn_tree_pred, squared=False).round(decimals=2)
MAE = mean_absolute_error(life_target_test, decsn_tree_pred).round(decimals=2)
MSE = mean_squared_error(life_target_test, decsn_tree_pred).round(decimals=2)
R2 = r2_score(life_target_test, decsn_tree_pred).round(decimals=2)
MAPE = mean_absolute_percentage_error(life_target_test, decsn_tree_pred).round(decimals=2)

# importing the calculated metrics variables into a DataFrame for further use:
dec_tree_data = {'Model': [regress2.__class__.__name__], 'Root Mean Squared Error': [RMSE],
                 'Mean Absolute Error': [MAE],'Mean Squared Error': [MSE],
                 'R-squared': [R2],'Mean Absolute Percentage Error': [MAPE]}

dec_tree_df = pd.DataFrame(dec_tree_data, columns=['Model', 'Root Mean Squared Error', 'Mean Absolute Error',
                                                   'Mean Squared Error', 'R-squared','Mean Absolute Percentage Error'])


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
print('Mean Absolute Percentage Error: : {:.2f}'.format(mean_absolute_percentage_error(life_target_test, rf_pred)))

# storing each of the computed Regression Metrices in a variable:
RMSE = mean_squared_error(life_target_test, rf_pred, squared=False).round(decimals=2)
MAE = mean_absolute_error(life_target_test, rf_pred).round(decimals=2)
MSE = mean_squared_error(life_target_test, rf_pred).round(decimals=2)
R2 = r2_score(life_target_test, rf_pred).round(decimals=2)
MAPE = mean_absolute_percentage_error(life_target_test, rf_pred).round(decimals=2)

rf_data = {'Model': [rf.__class__.__name__], 'Root Mean Squared Error': [RMSE], 'Mean Absolute Error': [MAE],
           'Mean Squared Error': [MSE], 'R-squared': [R2], 'Mean Absolute Percentage Error': [MAPE]}

# importing the calculated metrics variables into a DataFrame for further use:
rf_df = pd.DataFrame(rf_data, columns=['Model', 'Root Mean Squared Error', 'Mean Absolute Error',
                                       'Mean Squared Error', 'R-squared','Mean Absolute Percentage Error'])


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

# # plotting the graph at the bottom for Random Forest Regression Model:
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


## Step 9 - Performing Feature Selection on Life Expectancy data -
# A. Feature Selection (Embedded Method) using Lasso regression (L1 Regularization):
# importing the necessary libraries from scikit-learn:
from sklearn.linear_model import LassoCV

# instantiating our regressor for performing Lasso Cross-Validation:
reg = LassoCV()

# fitting our regressor to the training data:
reg.fit(life_feature_train,life_target_train)

# printing the best alpha and best score from Lasso CV using the alpha_ and score_ attribute :
print("Best alpha using LassoCV: %.2f" % reg.alpha_)
print("Best score using LassoCV: %.2f" % reg.score(life_feature_train, life_target_train)) # Returns the coefficient of determination

# creating a Pandas Series by extracting the coef_ attribute against each feature name as index:
coef = pd.Series(reg.coef_, index = life_feature.columns)

# printing the most important and least important feature counts:
print("Lasso picked " + str(sum(coef != 0)) + " features and eliminated the other " +  str(sum(coef == 0)) + " features")

# sorting the coefficients and visualizing them in a plot -
imp_coef = coef.sort_values()
# setting the figure size for the plot:
plt.figure(figsize=(8,5))
# plotting the coefficient counts as a horizontal bar plot:
imp_coef.plot(kind = "barh")
# adding the title and x-axis labels to the plot:
_ = plt.title("Feature Selection using Lasso Model", fontsize=14)
_ = plt.xlabel('Coefficients')
# displaying the plot:
plt.show()


# B. Feature Importance using Random Forest Regressor:
# Note - we had already run the Random Forest Regression model earlier so using feature_importances_ property here:
# creating a Pandas Series of important features:
importances_rf = pd.Series(rf.feature_importances_, index=life_feature.columns)

# sorting the feature Series created using the sort_values function :
sorted_importances_rf = importances_rf.sort_values()

# visualizing the importance of features as a horizontal bar plot:
sorted_importances_rf.plot(kind='barh',color='lightgreen')
# adding a title to the plot:
plt.title("Feature Importance using Random Forests", fontsize=14)
# displaying the plot:
plt.show()



## Step 10 - Implementing Gradient Boosting Algorithms to Improve Model Performance -
# A. Boosting through Gradient Boosting for Regression:
# importing the models and utility functions:
from sklearn.ensemble import GradientBoostingRegressor

# We have already split the dataset into 70% train and 30% test so do not repeat it.

# instantiating a gradient boosting regressor object:
# not specifying the subsample parameter defaults it to 1.0 hence, the model performs Gradient Boosting on data
gbt = GradientBoostingRegressor(n_estimators=400, max_depth=8,random_state=42)

# fitting 'gbt' to the training set of data:
gbt.fit(life_feature_train, life_target_train)

# predicting the test set labels:
gbt_pred = gbt.predict(life_feature_test)

# computing and printing the key Regression Metrices after performing Gradient Boosting:
rmse_test = mean_squared_error(life_target_test, gbt_pred)**(1/2)

print('Regression Metrics:',gbt.__class__.__name__,'\n')
print('Root Mean Squared Error after Gradient Boosting: {:.2f}'.format(rmse_test))
print('Mean Absolute Error after Gradient Boosting: {:.2f}'.format(mean_absolute_error(life_target_test, gbt_pred)))
print('Mean Squared Error after Gradient Boosting: {:.2f}'.format(mean_squared_error(life_target_test, gbt_pred)))
print('R_squared Score on actual vs. prediction: {:.2f}'.format(r2_score(life_target_test, gbt_pred)))
print('Mean Absolute Percentage Error after Gradient Boosting: {:.2f}'.format(mean_absolute_percentage_error(life_target_test, gbt_pred)))


# B. Boosting through Stochastic Gradient Boosting for Regression:
# We have already split the dataset into 70% train and 30% test so do not repeat it.

# instantiating a stochastic gradient boosting regressor object:
# specifying the subsample parameter < 1.0 leads to model performing Stochastic Gradient Boosting on the data
sgbt = GradientBoostingRegressor(n_estimators=400, max_depth=8, max_features=0.2,subsample=0.8,random_state=42)

# fitting 'sgbt' to the training set of data:
sgbt.fit(life_feature_train, life_target_train)

# predicting the test set labels:
sgbt_pred = sgbt.predict(life_feature_test)

# computing and printing the key Regression Metrices after performing Stochastic Gradient Boosting:
rmse_test = mean_squared_error(life_target_test, sgbt_pred)**(1/2)

print('Regression Metrics: StochasticGradientBoostingRegressor \n')
print('Root Mean Squared Error after Stochastic Gradient Boosting: {:.2f}'.format(rmse_test))
print('Mean Absolute Error after Stochastic Gradient Boosting: {:.2f}'.format(mean_absolute_error(life_target_test, sgbt_pred)))
print('Mean Squared Error after Stochastic Gradient Boosting: {:.2f}'.format(mean_squared_error(life_target_test, sgbt_pred)))
print('R_squared Score on actual vs. prediction: {:.2f}'.format(r2_score(life_target_test, sgbt_pred)))
print('Mean Absolute Percentage Error after Gradient Boosting: {:.2f}'.format(mean_absolute_percentage_error(life_target_test, sgbt_pred)))


# Visualizing the Actual vs. Predicted Life Expectancy values as Subplots from each of the Gradient Boosting Algorithms:
# setting up the matplotlib figure for subplots:
plt.figure(figsize=(18,6))
# adding a main title for all the subplots:
plt.suptitle('Visualizing the Actual vs. Predicted Life Expectancy values from the Gradient Boosting Algorithms')

# plotting the graph at the top for Gradient Boosting algorithm:
plt.subplot(2, 1, 1)
# creating the plot for Gradient Boosting algorithm as a scatter and line plot:
x_ax = range(len(life_target_test))
plt.scatter(x_ax, life_target_test, s=5, color="blue", label="actual")
plt.plot(x_ax, gbt_pred, lw=0.8, color="red", label="predicted")
# adding a title to the subplot:
plt.title(gbt.__class__.__name__)   # extracting a class name as a string to use as plot title
# adding a legend to the plot:
plt.legend()

# Plotting the graph at the bottom for Stochastic Gradient Boosting algorithm:
plt.subplot(2, 1, 2)
# creating the plot for Stochastic Gradient Boosting algorithm as a scatter and line plot:
x_ax = range(len(life_target_test))
plt.scatter(x_ax, life_target_test, s=5, color="blue", label="actual")
plt.plot(x_ax, sgbt_pred, lw=0.8, color="green", label="predicted")
#  adding a title to the subplot:
plt.title('StochasticGradientBoostingRegressor')
# adding a legend to the plot:
plt.legend()
# finally displaying the plot:
plt.show()


## Step 11 - Model Tuning using Grid Search Cross Validation -
# Step i) - Creating a Function to tune the hyperparameters of different models using GridSearchCV -
# importing the necessary library to perform GridSearchCV:
from sklearn.model_selection import GridSearchCV

# defining a Function model_tuning for all the Models -
def model_tuning(model, param_dict):
    """Function to tune the hyperparamters of a Model using Grid Search Cross Validation"""

    # inspecting the hyperparameters for the chosen Model:
    model.get_params()

    # defining the grid or combination of parameters that we want to test out on the chosen Model:
    model_grid = param_dict

    # instantiating a GridSearchCV object on the Model with scoring parameter set to r-squared and cross-validation folds set to 5:
    grid_model = GridSearchCV(estimator=model, param_grid=model_grid, cv=5,
                              scoring='r2')  # scoring means the metric to optimize

    # fitting the grid_model to the training set of data:
    grid_model.fit(life_feature_train, life_target_train)

    # extracting the best estimator, score and parameters from the model and printing them:
    print("Results from GridSearchCV on", model.__class__.__name__)

    # printing the estimator which gave highest score (or smallest loss if specified) on the left out data:
    print("Best Estimator:\n", grid_model.best_estimator_)

    # printing the mean cross-validated score of the best_estimator:
    print("\n Best Score:\n", grid_model.best_score_)

    # printing the parameter setting that gave the best results on the hold out data:
    print("\n Best Hyperparameters:\n", grid_model.best_params_)

    # extracting best model from the grid:
    best_model_grid = grid_model.best_estimator_

    # predicting the test set labels:
    y_pred = best_model_grid.predict(life_feature_test)

    # Computing & printing the key Regression Metrics after Tuning the model & predicting with the best parameters:
    rmse_test = mean_squared_error(life_target_test, y_pred) ** (1 / 2)

    print('\n Regression Metrices on Tuned Model:')
    # computing and printing the Root Mean Squared Error (RMSE) for the model:
    print('Root Mean Squared Error of Best Model: {:.2f}'.format(rmse_test))
    # computing and printing the Mean Absolute Error (MAE) for the model:
    print('Mean Absolute Error of Best Model: {:.2f}'.format(mean_absolute_error(life_target_test, y_pred)))
    # computing and printing the Mean Squared Error (MSE) for the model:
    print('Mean Squared Error of Best Model: {:.2f}'.format(mean_squared_error(life_target_test, y_pred)))
    # computing and printing the R-squared value for the model:
    print('R_squared Score on actual vs. prediction: {:.2f}'.format(r2_score(life_target_test, y_pred)))
    # computing and printing the Mean Absolute Percentage Error (MAPE) for the model:
    print('Mean Absolute Percentage Error of Model: {:.2f}'.format(
        mean_absolute_percentage_error(life_target_test, y_pred)))


# Step ii) Calling the model_tuning function with each model name and hyperparameter grid as arguments -
# A. Implementing GridSearchCV on Random Forest Regression Model:
# defining a grid of hyperparameters for Random Forest Regression Model:
params_rf = {'n_estimators': [300, 400, 500],
            'max_depth': [1, 4, 8],
            'min_samples_leaf': [0.1, 0.2],
            'max_features' : ['log2','sqrt']}
# calling the model_tuning function with requisite Model arguments:
model_tuning(RandomForestRegressor(random_state=42),params_rf)



# B. Implementing GridSearchCV on Ridge Regression Model:
# defining a grid of hyperparameters for Ridge Regression Model:
params_ridge = {'alpha': range(0,10),
            'max_iter': [10, 100, 1000]}
# calling the model_tuning function with requisite Model arguments:
model_tuning(Ridge(random_state=42),params_ridge)


# C. Implementing GridSearchCV on Gradient Boosting Regression Model:
# defining a grid of hyperparameters for Gradient Boosting Regression Model:
params_gbt = {'learning_rate': [0.01,0.02,0.03,0.04],
              'subsample':[0.8, 0.4, 0.2, 0.1],
              'n_estimators' : [100,400,800,1000],
            'max_depth': [4, 6, 8, 10]}

# calling the model_tuning function with requisite Model argument:
model_tuning(GradientBoostingRegressor(random_state=42),params_gbt)


# Step iii)  Using the Best Combination parameters from GridSearch Parameter Optimization & Running the Gradient Boosting Algorithm -
# We have already split the dataset into 70% train and 30% test.

# instantiating a Gradient Boosting Regressor, tuned_gbt with best combination parameters:
tuned_gbt = GradientBoostingRegressor(learning_rate=0.01, subsample=0.2, n_estimators= 800, max_depth=8, random_state=42)

# fitting tuned_gbt to the training set of Life expectancy dataset:
tuned_gbt.fit(life_feature_train, life_target_train)

# predicting the test set labels:
tuned_y_pred = tuned_gbt.predict(life_feature_test)

# comparing predicted Life Expectancy values with actual Life Expectancy values as a DataFrame:
# computing the difference between actual Life expectancies and predicted Life expectancies:
difference = life_target_test - tuned_y_pred
df = pd.DataFrame({'Actual':life_target_test, 'Predicted':tuned_y_pred, 'Residuals': difference})
# printing a snapshot of first 5 rows of the DataFrame:
print(df.head())

# visualizing the Actual vs. Predicted Life Expectancy values from the Tuned Gradient Boosting Regression Model:
# specifying the size of the plot:
plt.figure(figsize=(14,6))
# adding a title to the plot:
plt.title('Visualizing the Actual vs. Predicted Values for Tuned Gradient Boosting Regression Model', fontsize=14)
# creating the plot for Tuned Gradient Boosting algorithm as a scatter and line plot:
x_ax = range(len(life_target_test))
plt.scatter(x_ax, life_target_test, s=5, color="blue", label="actual")
plt.plot(x_ax, tuned_y_pred, lw=0.8, color="red", label="predicted")
# adding a legend to the plot:
plt.legend()
# displaying the plot:
plt.show()



## Step 12 - Comparing Models Performances through Visualization and Regression Metrices -
# Step i) Creating a Function to compute and store key Regression Metrices as a DataFrame -
# defining a Function compare_models for all the Models -
def compare_models(regressor):
    """Function to fit the Model to the training set, predict on testing data, compute the key Regression Metrices and
    store them in a DataFrame:"""

    # instantiating the regressor for the model:
    regress = regressor()

    # fitting the regressor to the training data using the fit method:
    regress.fit(life_feature_train, life_target_train)

    # predicting on the testing data using the predict method:
    life_pred = regress.predict(life_feature_test)

    # computing the key Regression metrices and storing each in a variable:
    RMSE = mean_squared_error(life_target_test, life_pred, squared=False).round(decimals=2)
    MAE = mean_absolute_error(life_target_test, life_pred).round(decimals=2)
    MSE = mean_squared_error(life_target_test, life_pred).round(decimals=2)
    R2 = r2_score(life_target_test, life_pred).round(decimals=2)
    MAPE = mean_absolute_percentage_error(life_target_test, life_pred).round(decimals=2)

    # storing all the variables of metrices as a Python dictionary, data:
    data = {'Model': [regress.__class__.__name__], 'Root Mean Squared Error': [RMSE], 'Mean Absolute Error': [MAE],
            'Mean Squared Error': [MSE],
            'R-squared': [R2], 'Mean Absolute Percentage Error': [MAPE]}
    # using this Python dictionary to create a Pandas DataFrame for all computed metrices from the Model:
    new_df = pd.DataFrame(data, columns=['Model', 'Root Mean Squared Error', 'Mean Absolute Error',
                                         'Mean Squared Error', 'R-squared', 'Mean Absolute Percentage Error'])
    # returning the Metrices' DataFrame created:
    return new_df


# Step ii) Running the compare_models function with different Models as argument and storing the output for each in a variable:
lin_reg = compare_models(LinearRegression)
ridg_reg = compare_models(Ridge)
lasso_reg = compare_models(Lasso)
gbt_reg = compare_models(GradientBoostingRegressor)


# Step iii) Concatenating the Metrices' DataFrames created from each Model into one:
# storing all the variables with Metrices' DataFrames from each model into a list:
metrices_df = [lin_reg,ridg_reg,lasso_reg,gbt_reg,rf_df,dec_tree_df]  # also combining metrices computed from the tree models
# concatenating metrices' list using the concat function:
combined_metrices = pd.concat(metrices_df)
# resetting the index of the combined DataFrame:
combined_metrices.reset_index()
# printing the combined DataFrame:
print(combined_metrices)


# Step iv) Visualization of Comparison of Metrices Computed from Different Models as Bar Plot and Table -
# setting the figure and axis object for the first (top) plot:
#fig, ax = plt.subplots(0,0)
# creating the plot of Regression Metrices computed from Models as a bar plot:
combined_metrices.plot(x='Model',y=['R-squared','Root Mean Squared Error','Mean Absolute Error', 'Mean Absolute Percentage Error'], kind='bar',figsize=(10,8))
# adding x-axis ticks and formatting them:
#plt.xticks(rotation = 45) # Rotates X-Axis Ticks by 45-degrees
# adding a title to the plot:
plt.title('Comparing Regression Metrices from Different Models')
plt.show()

# setting the figure and axis object for the second (bottom) plot:
#plt.subplots(0,1)
# specifying layout formats:
plt.axis('tight')
# setting the axes for table:
plt.axis('off')
# creating a Matplotlib table from the combined DataFrame for metrices computed:
tab = plt.table(cellText=combined_metrices.values,colLabels=combined_metrices.columns,loc="top",colColours =["yellow"] * 6,
              rowLoc='left',cellLoc='center')
# setting the fontsize of text in the table created:
tab.set_fontsize(15)
# setting the size of the table created:
tab.scale(1.6, 1.2)

# setting the plot's layout:
#fig.tight_layout()
# displaying the plot:
plt.show()



