#This is the main file for the Project

import pandas as pd
import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import streamlit as st
from PIL import Image


#read in data
s = pd.read_csv("social_media_usage.csv")

#creating function to clean linkedin user column
def clean_sm(x):
    x = np.where(x ==1, 1, 0)
    return(x)




##############################################
#Model creation
drop_list = s.index[np.where((s["age"] > 97) |
                            (s["income"] > 9) |
                            (s["educ2"] > 8) |
                            (s["par"] > 2) |
                            (s["marital"] > 6) |
                            (s["gender"] > 3))]
s.drop(drop_list, inplace = True)


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

"#### What is your income range? "
a = "Less than $10,000"
b = "10 to under $20,000"
c = "20 to under $30,000"
d = "30 to under $40,000"
e = "40 to under $50,000"
f = "50 to under $75,000"
g = "75 to under $100,000"
h = "100 to under $150,000"
i = "$150,000 or more"
j = "Don't know"
inc =  st.selectbox(label = "Select: ", 
                    options = ("Refuse to Answer",a,b,c,d,e,f,g,h,i))

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


#####################################
#Asking Education Level

"#### What is your highest level of education? "
a = "Less than high school (Grades 1-8 or no formal schooling)"
b = "High school incomplete (Grades 9-11 or Grade 12 with NO diploma)"
c = "High school graduate (Grade 12 with diploma or GED certificate)"
d = "Some college, no degree (includes some community college)"
e = "Two-year associate degree from a college or university"
f = "Four-year college or university degree/Bachelor’s degree (e.g., BS, BA, AB)"
g = "Some postgraduate or professional schooling, no postgraduate degree (e.g. some graduate school)"
h = "Postgraduate or professional degree, including master’s, doctorate, medical or law degree (e.g., MA, MS, PhD, MD, JD)"
j = "Don't know"
edu =  st.selectbox(label = "Select: ", 
                    options = ("Refuse to Answer",a,b,c,d,e,f,g,h))

if edu == a:
    educ2 = 1
elif edu == b:
    educ2 = 2
elif edu == c:
    educ2 = 3
elif edu == d:
    educ2 = 4
elif edu == e:
    educ2 = 5
elif edu == f:
    educ2 = 6
elif edu == g:
    educ2 = 7
elif edu == h:
    educ2 = 8
elif edu == j:
    educ2 = 98

#####################################
#asking parental status

"#### Are you a parent of a child under 18 living in your home?"
if st.checkbox("Yes"):
    par = 1
elif st.checkbox("No"):
    par = 2


#####################################
#asking marital status
"#### What is your current marital status?:"

a = "Married"
b = "Living with a Partner"
c = "Divorced"
d = "Separated"
e = "Widowed"
f = "Never been Married"
j = "Don't know"

mar =  st.selectbox(label = "Select the best that fits your current situation: ", 
                    options = ("Refuse to Answer",a,b,c,d,e,f))

if mar == a:
    marital = 1
elif mar == b:
    marital = 2
elif mar == c:
    marital = 3
elif mar == d:
    marital = 4
elif mar == e:
    marital = 5
elif mar == f:
    marital = 6
elif mar == j:
    marital = 98

#####################################
#asking gender

"#### What is your gender?"
if st.checkbox("Male"):
    gender = 1
elif st.checkbox("Female"):
    gender = 2
elif st.checkbox("Other"):
    gender = 3


#####################################
#asking age
"#### What is your age?"
age = st.slider("Age", min_value=18, max_value=97)



newdata = pd.DataFrame({
    "income" : [income],
    "educ2" : [educ2],
    "par" : [par],
    "marital" : [marital],
    "gender" : [gender],
    "age" : [age],
})

var_list = [income, educ2, par, marital, gender, age]
max_var = max(var_list)


if st.button("Submit"):
    if ((max_var > 97) | (par > 7) | (marital>6)):
        "## Please make sure all questions are answered before submission."
    else:
        probabilities = model1.predict_proba(newdata)
        p1 = probabilities[0,1]

        p1_str = "{:.2%}".format(p1)

        if p1 >= .5:
            "## You are a predicted Linkedin User!"
            im = Image.open("Li_logo.png")
        else:
            "## You are not a predicted Linkedin User!"
            im = Image.open("redx.png")

    
        st.write("The Percentage you are a user is: ", p1_str)
        im = im.resize((200,200))
        st.image(im)





