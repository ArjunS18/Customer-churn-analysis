Customer churn analysis using powerBi and Python

A brief overview of the project:
Problem Statement:
Customer churn is a significant issue for banks, as acquiring new customers can be up to five times more expensive than retaining existing ones. Additionally, existing customers can increase profits from 25% to 95%. Therefore, understanding and analyzing churn is crucial for banks to retain customers and increase customer lifetime value (CLV).

Churn analysis provides insights into potential avenues to improve services and even generate new revenue streams. By analyzing customer behavior, banks can identify the reasons behind customer churn and take necessary measures to prevent it.

Dataset:
The dataset contains more than 10,000 records collected over a period of 7 years across three countries, and was obtained from Kaggle for the Royal Bank of Canada.

Methodology - Python and PowerBI
- Conducted exploratory data analysis to identify null values in the dataset

- Identified 3618 accounts with 0 balance, which were also considered in the analysis for this problem statement

- Employed Random Forest Regressor and forward selection of variables to select top features with high impact on the target variable

- As the target variable 'Churn' was categorical, used Logistic Regression and Random Forest Classification methods to develop predictive models using sci-kit learn package

- Employed PowerBI to build interactive dashboards to provide insights on customers who are churning from the banks

- Utilized DAX functions in PowerBI to categorize and visualize the data in various charts such as line charts, ribbon charts, and pie charts.

Future Scope:
- Slice and dice the data to gain deeper insights and understanding of the churn reasons for the elder and female customer groups

- Implement other classification models such as Decision Trees, Gradient boosting and compare with existing models to find the most accurate model for the dataset

