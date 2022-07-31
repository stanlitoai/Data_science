import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv("heart-disease.csv")

df.head()
df.target.value_counts().plot(kind="bar")
df.sex.value_counts().plot(kind="bar")



"""
1. Problem definition
    predict heart disease

"""












"""
2. DATA

This is the data we`re using

"""
