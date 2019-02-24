import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

loans = pd.read_csv ()

loans.info

loans.describe()

loans.head()

#EXPLORATORY DATA ANALYSIS

#This is a histogram of two FICO distributions of fully paid loans- for both successfull and unsuccesful credit policy candidates
#Lendingclub.com has its own underwriting criteria, this is referred in the project as credit.policy
plt.figure(figsize = 10,6))
loans[loans['credit.policy']==1]['fico'].hist(alpha = 0.5,color = 'blue', bins = 30, label = 'Credit.Policy =1')
loans[loans['credit.policy']==0]['fico'].hist(alpha = 0.5,color = 'red', bins = 30, label = 'Credit.Policy =1')
plt.legend()
plt.xlabel('FICO')

#I am going to create the same histogram except for candidates who failed to pay their loans
plt.figure(figsize = 10,6))
loans[loans['not.fully.paid']==1]['fico'].hist(alpha = 0.5,color = 'blue', bins = 30, label = 'not.fully.paid =1')
loans[loans['not.fully.paid']==0]['fico'].hist(alpha = 0.5,color = 'red', bins = 30, label = 'not.fully.paid =1')
plt.legend()
plt.xlabel('FICO')

# I want to see the counts of loans categorized by the the reason of loan request
plt.figure(figsize = (11,7))
sns.countplot( x= 'purpose', hue = 'not.fully.paid', data = loans, palette = 'Set1')

# I want to see the tred between the FICO score and the interest rate
sns.jointplot(x='fico',y = 'int.rate',data = loans, color = 'purple')

#Now, I want to see if there are different trends for credit.policy and not.fully.paid
plt.figure(figsize=(11,7))
sns.lmplot(y='int.rate', x= 'fico', data =  loans, hue = 'credit.policy', col = 'not.fully.paid', palette = 'Set1')

#The purpose column in the dataset is categorical. I am going to transform it into a dummy variable before implementing the model on it
categorical_data = ['purpose']
final_data = pd.get_dummies(loans, columns - cat_feats, drop_first = True)

#Now I am going to check if the data was transformed
final_data.info()

#Splitting the data into training and testing
from sklearn.model_selection import train_test_split
X = final_data.drop('not.fully.paid', axis =1)
y = final_data['not.fully.paid']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size =0.3, random_state = 101)

from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier
dtree.fit(X_train,y_train)

#Predictions and Evaluation of Decision Tree
#I am going to create predictions from the test set and create a classification replort and confusion_matrix
predictions = dtree.predict(X_test)
from sklearn.metrics import classification_report,(y_test,predicions))
print(classification_report(y_test, predictions))

#I would like to also compare the model with Random Forrest Model

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestCLassifier(n_estimator = 600)
rfc.fit(X_train,y_train)


predictions = rfc.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test, predictions))

print(confusion_matrix(y_test, predictions))

#S
