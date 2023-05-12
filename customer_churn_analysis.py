"""**Problem Statement:**
It is important for banks to identify customers who are likely to churn and take proactive measures to retain them. By building a predictive **logistic regression** model that can identify customers who are likely to churn, banks can take appropriate measures to retain them and reduce customer churn. This can help the bank to improve customer satisfaction, increase customer retention rates, and ultimately increase profitability. Additionally, by understanding the factors that contribute to customer churn, banks can make strategic decisions to improve customer satisfaction and loyalty.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
!pip install mlxtend
import sys
import joblib
sys.modules['sklearn.externals.joblib'] = joblib
from mlxtend.feature_selection import SequentialFeatureSelector as sfs
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from google.colab import files
customer_churn = files.upload()

#read csv file
df = pd.read_csv('Bank_Churn.csv')
df_bkup = df.copy(deep=True)
df.head(5)

#Exploratory data analysis
print('Number of rows in dataframe: {}'.format(df.shape[0]))
print('Number of features in dataframe: {}'.format(df.shape[1]))

#Find the data type of each feature of the dataframe
df.dtypes.to_frame('Datatype of each feature').T

#Count the number of NULL values in all features
df.isnull().sum().to_frame('Count of missing values')

# Create a correlation matrix to find out the highest correlation variable with churn.
plt.figure(figsize=(20, 10))
heatmap = sns.heatmap(df.corr(), annot=True, cmap='Blues', square=True)
heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':12}, pad=12);

#print(df.corr())

# Create list of features
features = df.keys().drop('Exited')
features

# Check central tendencies, grouped by churn
df.groupby('Exited')[features].mean().style.background_gradient(cmap = "YlOrRd")

# Encode the categorical variables into numeric, so they can be inputted into the model
le = LabelEncoder()
df['gender'] = le.fit_transform(df['gender'])
df['country'] = le.fit_transform(df['country'])

# use forward selection to find features impacting target variable
X = df.drop(['RowNumber', 'Exited', 'CustomerId', 'Bank DOJ' ], axis=1)
y = df['Exited']
X.shape, y.shape

#split the data into test and train
#X = df[feature_names]
Y = df['Exited']
X_train, X_test, y_train, y_test = train_test_split( 
    X, Y, test_size = 0.25, random_state = 1)

null_acc = y_test.describe()[3]/y_test.describe()[0]
print(null_acc)

y_test.value_counts()

#use forward selection to select variables that have impact on the target variable (churn)
lreg = LogisticRegression()
sfs1 = sfs(lreg, k_features=3, forward=True, verbose=2, scoring='neg_mean_squared_error')
sfs1 = sfs1.fit(X, y)

feature_names = list(sfs1.k_feature_names_)
print(feature_names)

from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
sel = SelectFromModel(RandomForestRegressor(n_estimators = 100))
sel.fit(X_train, y_train)
random_forest_output = list(sel.get_support())
colnames = X.columns.tolist()
for i in range(len(colnames)):
  print(colnames[i], random_forest_output[i])

# Access the underlying RandomForestRegressor model
rf_model = sel.estimator_

# Print feature importance values
importances = rf_model.feature_importances_
print(type(importances))
#print(importances)
colnames = X.columns.tolist()
for i in range(len(colnames)):
    print('features: {0} - {1}'.format(colnames[i], np.round(importances[i], 4)))

# use forward selection to find features impacting target variable
X = df.drop(['RowNumber', 'Exited', 'CustomerId', 'Bank DOJ', 'Tenure', 'HasCrCard', 'NumOfProducts', 'GeographyID', 'GenderID', 'EstimatedSalary', 'IsActiveMember' ], axis=1)
y = df['Exited']
X.shape, y.shape

#split the data into test and train
#X = df[feature_names]
Y = df['Exited']
X_train, X_test, y_train, y_test = train_test_split( 
    X, Y, test_size = 0.25, random_state = 1)

#find churn rate for X_train, count the people in each category - age, credit score 
#present all numbers in percentage
#Y_train.coun
y_train.value_counts()

#As balance, credit score mean values is higher than other features,
#we have to scale the features. If not, balance and credit score will
#have more impact on the logistic regression model
sc = StandardScaler()
xtrain = sc.fit_transform(X_train)
xtest = sc.transform(X_test)  
print (xtrain[0:10, :])

classifier = LogisticRegression(random_state = 1)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print ("Confusion Matrix : \n", cm)

#find the accuracy of the model
#print(y_test)
#print(y_pred)
print ("Accuracy : ", accuracy_score(y_test, y_pred))

# Fitting Random Forest Regression to the dataset
from sklearn.ensemble import RandomForestClassifier
regressor = RandomForestClassifier(n_estimators = 5, random_state = 0)
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)
print ("Accuracy : ", accuracy_score(y_test, y_pred))

