## UCDPA2_SukritiShukla

# **Understanding and Predicting Life Expectancy using Machine Learning in Python**

## (Mastering Machine Learning using Python)
UCD Professional Academy - Speciaist Certificate in Data Analytics Essentials **Final Project**

Prepared by - [Sukriti Shukla](https://www.linkedin.com/in/sukriti-shukla-3989a819/)

![https://images.pexels.com/photos/7615460/pexels-photo-7615460.jpeg?auto=compress&cs=tinysrgb&dpr=2&h=650&w=940](https://images.pexels.com/photos/7615460/pexels-photo-7615460.jpeg?auto=compress&cs=tinysrgb&dpr=2&h=650&w=940)

Photo by Nataliya Vaitkevich from Pexels

## Introduction 

According to Wikipedia, [Life Expectancy](https://en.wikipedia.org/wiki/Life_expectancy), is a statistical measure of the average time an organism is expected to live, based on the year of its birth, its current age, and other demographic factors including biological sex. The most used measure is life expectancy at birth (LEB). Life expectancy is a key metric for assessing population health and telling the average age of death in a population. It is one of the factors in measuring the Human Development Index (HDI) of each nation along with adult literacy, education, and standard of living. Understanding the key forms of life expectancy like Health Life Expectancy at birth (HALE) and Life Expectancy at birth (LEB) have been important tools not only for nations in improving their health care and social security schemes but have also helped corporates like insurance providers in assessing their cash outflows. But even today, there are great variations in life expectancy between different parts of the world, mostly caused by differences in public health, medical care, and diet.

The purpose of this project was to understand what Life Expectancy at birth (LEB) is and how is it distributed across the world. Aggregation and groupings were used to estimate the trends in Life expectancies over a period, and what factors influence it. Machine Learning (ML) algorithms were used to analyze Life Expectancy and its relationship with other factors like BMI, Alcohol consumption, Schooling, Adult Mortality etc. Supervised ML techniques like Linear Regression were used to make predictions about life expectancies. To evaluate different regression models, error metrics like Mean Squared Error, Mean Absolute Percentage Error were used as they summarized how close predictions were to their expected values along with the coefficient of determination (R-squared) score to compare model performance. 

Finally, scikit-learn was integrated with Matplotlib and Seaborn libraries to visualize these metrics, model coefficients and forecast errors from different models and choose the model that best predicts Life expectancies from the dataset.


## Data Sources and Preparation 

I retrieved data from different open sources in varied formats like csv files, REST APIs etc. I processed them using Python in-built functions and libraries such as Pandas for manipulation and structuring them as DataFrames. Numpy functions along with statistical analysis techniques like scaling were used to align the ranges of different factors. Regular Expression libraries like regex were used for pattern recognition and data slicing. Once the data was cleaned & formatted it was used to create a series of visualizations and draw insights.

For this project I primarily used two sources of data in my project - 
1. Kaggle Public Dataset - [Life Expectancy (WHO)](https://www.kaggle.com/kumarajarshi/life-expectancy-who) – dataset shared on Kaggle - csv file (*Life Expectancy Data.csv*)
2. Global Health Observatory by World Health Organization (WHO) [(GHO) Portal](https://www.who.int/data/gho) - The GHO portal provides a simple query interface to the World Health Organization's data and statistics content, using OData (Open Data Protocol).GHO currently has two Data APIs – GHO OData API and Athena API.For my project, I used [GHO OData API](https://www.who.int/data/gho/info/gho-odata-api) to query data for my analysis & visualizations.

As my dataset contained several independent factors such as adult mortality, infant deaths, GDP levels etc. influencing Life Expectancy I used Regression-based models (tree-models like Decision Trees, Random Forests and regression techniques like Linear Regression, Lasso Regression etc) to make predictions. Supervised Machine Learning algorithms from scikit-learn (also called sklearn) library were used to fit models to the training set and make predictions on the testing set. I compared models and tuned weaker models to improve their performance using Boosting algorithms and Hyperparameter tuning to optimize prediction accuracies using Grid Search. Gradient Boosting were used to improve the prediction power of the model by strengthening a weak model. I also tried to implement Feature Selection and Feature Importance using a tree model and a regularization technique to understand which factors from the Life expectancy dataset were the most relevant. I also used the Yellowbrick library, a diagnostic visualization platform for machine learning that allows data scientists to steer the model selection process. Its scikit-learn API with new core object: the Visualizer was used to make some key graphs for tuned models.

  I tried to understand variations in Life Expectancy at birth (LEB) between Genders (Male vs. Female) through charts and metrices over a period of over 20 years (2000-19) and studied how the distribution of LEB changed over time for various World Bank Income Group economies. Both the predictions made using Machine learning tools in Python and basic Seaborn and Matplotlib visualizations helped me in learning more about what factors have contributed to the Life Expectancy changes over the last 15-20 years across the world and which of them could be focused on more by governments to improve their health conditions and overall quality of life by reducing mortality rates and disease prevention.

## Visualizations & Insights Drawn 





## Conclusion


