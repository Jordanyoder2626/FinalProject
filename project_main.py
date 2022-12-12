#This is the main file for the Project

import pandas as pd
import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import streamlit as st


#read in data
s = pd.read_csv("social_media_usage.csv")

#creating and testing function for use on dataset
def clean_sm(x):
    x = np.where(x ==1, 1, 0)
    return(x)




##############################################
#Model creation

s['sm_li'] = clean_sm(s['web1h'])
s = s.astype('category')
s['age'] = s['age'].astype('int64')


#web1h is linkedin
ss = s[['income', 'educ2', 'par', 
    'marital', 'gender', 'age', 'sm_li']]



#setting target variable
y = ss['sm_li']

#features used for prediction
X = ss[['income', 'educ2', 'par', 
    'marital', 'gender', 'age']]


# X_train - training data features for prediction on test set
# X_test - testing data features for prediction on test set
# y_train - training data target variable we are predicting - used to train the model to make these predictions
# y_test - final set of data containing target variable - unseen, so it is used for model performance

# splitting data into training and test
# all variables containing train = 80% of dataset
# all variables containing test = 20% of dataset
X_train, X_test, y_train, y_test = train_test_split(X,
                                                   y,
                                                    stratify=y,       
                                                    test_size=0.2,    
                                                    random_state=18)


#Creating logistic regression and fitting model to training data
model1 = LogisticRegression(class_weight = "balanced")
model1.fit(X_train, y_train)

#predicting based on testing data
y_pred = model1.predict(X_test)

income = 99
educ2 = 99
par = 8
marital = 99
gender = 99
age = 98


"# Linkedin User Projection"

"#### What is your income range: "
a = "Less than $10,000"
b = "10 to under $20,000"
c = "20 to under $30,000"
d = "30 to under $40,000"
e = "40 to under $50,000"
f = "50 to under $75,000"
g = "75 to under $100,000"
h = "100 to under $150,000"
i = "$150,000 or more?"
j = "Don't know"
inc =  st.selectbox(label = "Select: ", 
                    options = ("Refuse to Answer",a,b,c,d,e,f,g,h,i,j))

if inc == a:
    income = 1
elif inc == b:
    income = 2
elif inc == c:
    income = 3
elif inc == d:
    income = 4
elif inc == e:
    income = 5
elif inc == f:
    income = 6
elif inc == g:
    income = 7
elif inc == h:
    income = 8
elif inc == i:
    income = 9
elif inc == j:
    income = 98
