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
2. Global Health Observatory by World Health Organization (WHO) [(GHO) Portal](https://www.who.int/data/gho) - The GHO portal provides a simple query interface to the World Health Organization's data and statistics content, using OData (Open Data Protocol).

GHO currently has two Data APIs – GHO OData API and Athena API.For my project, I used [GHO OData API](https://www.who.int/data/gho/info/gho-odata-api) to query data for my analysis & visualizations.

As my dataset contained several independent factors such as adult mortality, infant deaths, GDP levels etc. influencing Life Expectancy I used Regression-based models (tree-models like Decision Trees, Random Forests and regression techniques like Linear Regression, Lasso Regression etc) to make predictions. Supervised Machine Learning algorithms from scikit-learn (also called sklearn) library were used to fit models to the training set and make predictions on the testing set. I compared models and tuned weaker models to improve their performance using Boosting algorithms and Hyperparameter tuning to optimize prediction accuracies using Grid Search. I tried to implement Feature Selection and Feature Importance using a tree model and a regularization technique to understand which factors from the Life expectancy dataset were the most relevant. I also used the Yellowbrick library, a diagnostic visualization platform for machine learning that allows data scientists to steer the model selection process. Its scikit-learn API with new core object: the Visualizer was used to make some key graphs for tuned models.

  I tried to understand variations in Life Expectancy at birth (LEB) between Genders (Male vs. Female) through charts and metrices over a period of over 20 years (2000-19) and studied how the distribution of LEB changed over time for various World Bank Income Group economies. Both the predictions made using Machine learning tools in Python and basic Seaborn and Matplotlib visualizations helped me in learning more about what factors have contributed to the Life Expectancy changes over the last 15-20 years across the world and which of them could be focused on more by governments to improve their health conditions and overall quality of life by reducing mortality rates and disease prevention.
  
To summarize, I used the datasets in the following different formats to work on this project – 
* REST API (open source) 
* CSV file

## Visualizations & Insights Drawn 

For this project, I made use of Supervised Machine Learning techniques for regression-based data in Python to predict Life Expectancy. The different models were compared and weaker models tuned to improve their performance using Boosting algorithms and Hyperparameter tuning to optimize their predictions. I tried to understand through charts and metrices variations in Life Expectancy at birth (LEB) between Genders (Male vs. Female) over a period of over 20 years (2000-19) and studied how the distribution of LEB changed over time for various World Bank Income Group economies. 

Visualization of predictions were made using Seaborn and Matplotlib libraries. Visualizations and insights drawn from them helped me learn more about what factors have contributed to the Life Expectancy changes over time. It also explained which factors could be focused on more by governments to improve their health conditions and overall quality of life by reducing mortality rates and disease prevention.


 #### Chart 1 - Visualizing the Correlation among different Life Expectancy related factors as a Heatmap –

The Heatmap displayed the correlation among all the factors affecting Life Expectancy. The legend showed darker colors as having high negative correlation, while the lighter colors showed positive correlation. It was useful to understand how each of the factors like BMI or Alcohol consumption related to Life expectancy and how each would help understand the predictions made by Machine learning algorithms in the later stages of analysis.

![Chart 2](https://user-images.githubusercontent.com/50409210/126188793-86cebc82-4cdc-4b03-98c7-57912bd68316.PNG)

The following set of ***insights*** could be drawn from the above visualization: -
* BMI, Income composition of resources and Schooling are the three most positively correlated to Life Expectancy while Adult Mortality and HIV/AIDS are the two most negatively     correlated. 
* There exists a high correlation between some factors themselves like between thinness of 5–9 year-old and thinness 1-19 year-old. Similarly, high correlation exists between     Population and infant deaths. This multi-collinearity attribute among variables can be an influencer in some model performances during machine learning.


 #### Chart 2 - Visualizing the Life Expectancy trends for Developed vs. Developing countries as Boxplots –

This chart had been plotted to show the distribution of Life expectancy values between Developed and Developing countries over a span of 16 years (from 2000-15). It studied if there was any trend in the life expectancy values over a period of 20 years. Time-series based analysis was important to understand if time played any role in the trends and if better technology, more resources influenced life expectancy values or not.

![Chart 4](https://user-images.githubusercontent.com/50409210/126189210-a03e5492-1f5a-4e70-aaeb-015f6f6f4552.PNG)


***Insights drawn*** - Developing economies lagged the developed countries throughout this period. However, there was a slight upward trend (except for 2014) in the developing economies’ distributions. From 2005 onwards, the median value of life expectancies, showed an improvement in those nations as well.


 #### Chart 3 - Plotting the Distribution of Life Expectancy at birth (years) by Gender during 2000-19 –

Seaborn violin plots were used to depict gender-wise of life expectancies over time. The x-axis showed the different years while the y-axis the life expectancy at birth values. The plot was segregated by Gender using the hue parameter in Seaborn. Thus, for each of the years we had 3 set of violin plots was made, one for each Gender and the third one showing both genders.

![Chart 20](https://user-images.githubusercontent.com/50409210/126189815-39352f09-1552-4621-ae61-072204e258f9.PNG)

***Insights from the plot*** - A violin plot was chosen here as it visualizes data distributions very clearly. The hue argument helped in clearly splitting the plot by Sex. As can be seen above, in each of the years, the Females had a higher median life expectancy value than males which meant that a female newborn had a longer expected life than a male newborn from 2000-19. The year 2010 has a much larger distribution for all the three Genders than other years. This might be due to some factor specific to that year.


 #### Chart 4 - Distribution of Life Expectancy at birth (years) by World Bank Income Groups during 2000-19 –

The World Bank classifies world’s economies into 4 income groups – high, upper-middle, lower-middle, low. As the API data rows had Country_Code starting with WB_ but missing Region Names, Country_Name and Region_Code, I used Regular Expressions (regex) library in Python to slice Country_Code matching to patterns starting with WB_. All matches were stored in an empty list and then used to slice the complete DataFrame to get only World Bank Income Groups-specific data.

Thereafter, data for these World Bank income groups was sliced and used to create a boxplot. The Boxplot showed years ranging from 2000-19 on the x-axis while Life expectancies at birth on the y-axis. The plot was segregated by World Bank Income Groups using the hue parameter of the plot. Thus, for each of the years we had 4 set of box plots, one for each World Bank Income group.

![Chart 21](https://user-images.githubusercontent.com/50409210/126190479-3ce905ef-d070-4032-bdb8-5b0858f9de89.PNG)


***Insights drawn from boxplots*** - The boxplots clearly show the distribution of life expectancies across income groups. The Low Income group economies had the lowest distribution of life expectancy values for all the years while the High Income groups had the highest distributions. Also, the interquartile ranges for Upper-Middle and High Income groups are more spread or greater than those for Low Income or Lower-Middle Income groups.


### Visualizations from Running Machine Learning Algorithms on Life Expectancy data -
 #### Chart 5 - Visualizing the Coefficients generated from Linear Regression Model - 

![Chart 6](https://user-images.githubusercontent.com/50409210/126191074-a70dbff2-838d-4dee-8005-180146d5af86.PNG)


The sign of a regression coefficient shows if there is a positive or negative correlation between some independent variables and the Life expectancy (the dependent variable). For example, here there is a negative correlation between adult mortality and life expectancy or with under-five deaths. A positive coefficient indicates that as the value of the independent variable increases, the mean of the dependent variable also tends to increase like for Income composition of resources or Schooling. Both these factors tend to improve average life expectancy to some extent. This can also be said about developed countries where both these variables are high owing to high life expectancies.

Similar charts were made for Lasso and Ridge regularization models.


 #### Chart 6 - Visualizing Feature Selection using Lasso Regression - 

Lasso regression can be used to select important features of the life expectancy dataset. If the feature is irrelevant, Lasso penalizes its coefficient and shrinks it to 0. Hence, the features with coefficient = 0 are removed while features whose coefficients are not shrunk to 0 are 'selected' by the LASSO algorithm. The below plot tried to understand the most relevant factors with respect to the life expectancy dataset.

![Chart 13](https://user-images.githubusercontent.com/50409210/126191693-a389932a-7ec2-463e-addf-4baab78f0fb0.PNG)


 #### Chart 7 - Comparison of Regression Metrices Computed from Different Machine Learning Algorithms as a Horizontal Bar Plot and a Table –

![Chart 17](https://user-images.githubusercontent.com/50409210/126191965-e281f2f0-b828-4483-bf73-d9947ea673c1.PNG)

The above plot, tried to combine the results from different Machine learning algorithms into one plot. It contained 2 subplots – 
* a Pandas horizontal bar plot at the top and 
* a Matplotlib table below it

The multiple horizontal bar plot showed each of the six models used to make predictions on the x-axis while the y-axis showed the metrics values. 

For each model, the above 5 metrices were displayed as bars lengths. The Matplotlib table below the bar plot also displayed these computations for each model. Both these visuals helped in easy comparison of model performance. Although the models compared through this chart were not tuned models using GridSearchCV these comparisons still helped me get an idea which model would perform better than others if tuned even further with the best combination of hyperparameters.


The following set of ***insights*** could be drawn from this comaprison - 
It was found that the Gradient Boosting Regression Algorithm works the best for the Life Expectancy dataset.
* It yielded the least error values of RMSE, MAE and MSE while the R-squared value achieved was also the highest. 
* It also resulted in incredibly low forecast error (MAPE) which showed that it fit the data quite well and was fairly accurate in making predictions about Life Expectancy values (however,these metric values were still not from Tuned Gradient Boosting Algorithm).


 #### Chart 8 - Visualizing Gradient Boosting Algorithm’s Performance on Life Expectancy data using Yellowbrick Regression Visualizers –

These set of visualization were made using Yellowbrick an open source scikit-learn API with visual analysis and diagnostic tools. I used its new core object: the Visualizer to fit the Gradient Boosting model and evaluate its performance on life expectancy dataset.

Its Regression Visualizers were used to make 2 regression evaluations for the Gradient Boosting model: 
* Residual plot – plotting the difference between expected and actual values. 
* Prediction Error – plotting the expected vs. actual values in model space.

![Chart 18](https://user-images.githubusercontent.com/50409210/126192771-fa2db671-41db-4bb7-b9d7-032523ab1a48.PNG)

For the Gradient Boosting algorithm, the Residual plot displayed the variance of error. There is also a histogram on the right showing the same. It shows that the R-squared for testing data from our life expectancy dataset is 0.935 or 93.5% which is quite like the scikit-learn based predictions we made earlier.

![Chart 19](https://user-images.githubusercontent.com/50409210/126192786-91f58c48-71d3-4c88-acf9-83eef0e48708.PNG)

For the Gradient Boosting algorithm on life expectancy data, we can see from the Predction Error plot that there is little error for the model and the R-squared value also shows a value of 0.935 or 93.5% implying that around 93.5% of life expectancy data fits the Gradient Boosting algorithm.




## Conclusion

#### Conclusions from Machine Learning Predictions about Life Expectancy Dataset –
Based on the visualizations and comparative analysis of different models tried in this project, I found that the tuned Gradient boosting algorithm (more precisely the tuned Stochastic Gradient Boosting algorithm, as subsample < 1.0) was most appropriately able to make predictions on the Life Expectancy dataset from Kaggle. 

![Chart 16](https://user-images.githubusercontent.com/50409210/126193169-224034bd-a56c-4b86-931b-0728d14814f2.PNG)

* This model provided the least error values, gave the highest R-squared value of around 93-94%, and achieved the least forecast error (MAPE) of around 3%, out of all the models   tried even without applying model tuning (as shown in the snapshots below).

* Its plot of actual values against predicted life expectancies as shown above (and from Yellowbrick API) along with the below snapshot of Actual vs. Predicted Life Expectancies   also gave little differences or Residuals. Apart from Gradient Boosting algorithm, the Linear and Ridge regression models also performed quite well on the Life Expectancy       dataset from Kaggle in making predictions


#### Learning Takeaways from the Project - 

This project helped me get a comprehensive view about the Life Expectancy in general. I was able to analyze what factors were related to it and which influence it the most. I could get insights about how life expectancies differed between developed and developing countries. 

By visualizing the distribution of life expectancy from 2000-19 I was able to see the trends and segregation across genders and World Bank Income groups. It also showed that females have higher life expectancies than males and what factors might have led to richer economies like Europe or Americas having better life expectancies than poorer ones like Africa. 

While implementing machine learning algorithms on the data I could assess how related factors affect life expectancies and how factors like schooling can improve them while infant deaths can make them worse. Governments and global bodies like the WHO, the World Bank or UNDP have been tracking life expectancies across the world to undertake important demographic and health-based research but in these tough times of a pandemic hitting the entire world predictions like these using Machine learning techniques can be even more useful to forecast the inequalities in life expectancies across the globe and which sections of the world might be the worst affected.
