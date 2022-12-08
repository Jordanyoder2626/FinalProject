#This is the main file for the Project

import pandas as pd
import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

#read in data
s = pd.read_csv("social_media_usage.csv")

#creating and testing function for use on dataset
def clean_sm(x):
    x = np.where(x ==1, 1, 0)
    return(x)

toy = pd.DataFrame({
    "Name": ["dog", "car", "bear"],
    "Cost": [10, 1, 15]
})

toy['1$ section'] = clean_sm(toy['Cost'])

print(toy)


##############################################
#Model creation


#web1h is linkedin
ss = s[["web1a","web1b","web1c","web1d",
    "web1e","web1f","web1g","web1h","web1i",
    "web1j","web1k", "income", "educ2", "par", 
    "marital", "gender", "age"]]

ss["sm_li"] = clean_sm(ss["web1h"])

#setting target variable
y = ss["sum_li"]

#features used for prediction
X = ss[["web1a","web1b","web1c","web1d",
    "web1e","web1f","web1g","web1i",
    "web1j","web1k", "income", "educ2", "par", 
    "marital", "gender", "age"]]


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


ss.lm_si.value_counts()

