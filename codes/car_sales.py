import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv("car-sales.csv")

df.head()


df.describe()


df.dtypes
df.info()

df.mean()

df["Odometer (KM)"].hist()


df["Price"] = df["Price"].str.replace('[\$\,\.]', '').astype(int)

df.head()


df["Price"].plot()

df["Make"].str.lower()


df_missing = pd.read_csv("car-sales-missing-data.csv")
df_missing

df_missing["Odometer"].fillna(df_missing["Odometer"].mean(),
                              inplace=True)


df = pd.read_csv("dog-vision-full-model-predictions-with-mobilenetV2.csv")

df.head()

import numpy as np

random_array = np.random.randint(0, 10, size=(4, 6))

random_array




#####DOT product

np.random.seed(0)

sales_amount = np.random.randint(20, size=(5, 3))
sales_amount


weekly_sales = pd.DataFrame(sales_amount,
                            index=["Mon", "Tue", "Wed", "Thurs", "Fri"],
                            columns=["Almond butter", "Peanut butter", "Cashew butter"])




weekly_sales


prices = np.array([10, 8, 12])

butter_prices = pd.DataFrame(prices.reshape(1, 3),
                             index=["Prices"],
                             columns=["Almond butter", "Peanut butter", "Cashew butter"])


butter_prices


###Using dot

total_sales = prices.dot(sales_amount.T)
total_sales 

weekly_sales["Total_sales"] = total_sales

weekly_sales







































