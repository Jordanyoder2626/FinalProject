#This is the main file for the Project

import pandas as pd
import os
import numpy as np

s = pd.read_csv("social_media_usage.csv")


def clean_sm(x):
    x = np.where(x ==1, 1, 0)
    return(x)

toy = pd.DataFrame({
    "Name": ["dog", "car", "bear"],
    "Cost": [10, 1, 15]
})

toy['1$ section'] = clean_sm(toy['Cost'])

print(toy)

ss = s[["web1a","web1b","web1c","web1d",
    "web1e","web1f","web1g","web1h","web1i",
    "web1j","web1k", "income", "educ2", "par", 
    "marital", "gender", "age"]]


