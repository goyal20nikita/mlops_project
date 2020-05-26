#!/usr/bin/env python
# coding: utf-8


import sys
import pandas as pd
dataset=pd.read_csv("/pyfile/Social_Network_Ads.csv")
dataset
dataset.info()
dataset.head()
dataset.columns
X=dataset[['Age','EstimatedSalary']]
y=dataset["Purchased"]
import seaborn as sns
sns.set()
sns.scatterplot(x="Age",y="EstimatedSalary",data=dataset,hue="Purchased")
type(X)
X=X.values
type(X)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
X_train
X_test
from sklearn.neighbors import KNeighborsClassifier
x=int(sys.argv[1])
if(x==0):
    model=KNeighborsClassifier(n_neighbors=3)
    model.fit(X_train,y_train)
    model.predict([[61,200000]])
    y_pred=model.predict(X_test)
    y_pred
    y_test
    from sklearn.metrics import confusion_matrix
    confusion_matrix(y_test,y_pred)
    from sklearn.metrics import accuracy_score
    accuracy=accuracy_score(y_test,y_pred)
    print(accuracy)

if(x==1):
  for i in range(4,10):
    model=KNeighborsClassifier(n_neighbors=i)
    model.fit(X_train,y_train)
    model.predict([[61,200000]])
    y_pred=model.predict(X_test)
    y_pred
    y_test
    from sklearn.metrics import confusion_matrix
    confusion_matrix(y_test,y_pred)
    from sklearn.metrics import accuracy_score
    accuracy=accuracy_score(y_test,y_pred)
    print(accuracy)
    if (accuracy>0.80):
         break




