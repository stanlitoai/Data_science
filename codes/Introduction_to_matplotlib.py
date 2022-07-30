##introduction to matplotlib

%matplotlib inline
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


plt.plot()


x =np.random.randn(1000)
fig, ax = plt.subplots()
ax.hist(x);


"""
#######################################

MAKING MULTIPLE PLOTS

"""
##Subplot option1

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2,
                                             ncols=2,
                                             figsize=(10, 6))

ax1.plot()
ax2.scatter()
ax3.plot()
ax4.hist()



####Subplot option2

fig, ax = plt.subplots(nrows=2,
                       ncols=2,
                       figsize=(10, 6))
#Plot to each different index

ax[0, 0].plot()
ax[0, 1].scatter()
ax[1, 0].plot()
ax[1, 1].hist()



import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv("car-sales.csv")

df.head()


ts = pd.Series(np.random.randn(1000),
               index=pd.date_range('1/1/2022', periods=1000))

ts = ts.cumsum()
ts.plot()

df["Price"] = df["Price"].str.replace('[\$\,\.]', '')

df.head()

type(df["Price"][0])

#Remove last two zeros
df["Price"] = df["Price"].str[:-2]




df["Sale Date"] = pd.date_range('1/1/2022', periods=len(df))

df["Total sales"] = df["Price"].astype(int).cumsum()

df

#Lets plot the total sales

df.plot(x="Sale Date", y="Total sales")

##Plot scatter plot with price column
df.plot(x="Odometer (KM)", y="Price", kind="scatter");


df.plot.bar();


df.plot(x="Make", y="Odometer (KM)", kind="bar");

df["Price"] = df["Price"].astype(int)
df.plot(x="Make", y="Price", kind="bar", grid=True);



df_hrt = pd.read_csv("heart-disease.csv")

df_hrt.head(20)

#Creating a histogram of age
df_hrt["age"].plot.hist();




df_hrt.plot.hist(figsize=(),subplots=True);




"""
Which one should you use? (pyplot vs matplotlib OO method)
*When plotting something quickly, okay to usee the pyplot method
*When plotting something more advanced, use the OO method

"""

over_50 = df_hrt[df_hrt["age"] > 50]
over_50.head()

over_50.plot(kind="scatter",
             x="age",
             y="chol",
             c="target");





##OO method mixed with plt

fig, ax = plt.subplots(figsize=(10, 5))
over_50.plot(kind="scatter",
             x="age",
             y="chol",
             c="target",
             ax=ax,
             grid=True);

ax.set_xlim([45, 100])



##OO method from scratch

plt.style.use("seaborn")
fig, ax = plt.subplots(figsize=(10, 5))
scatter= ax.scatter(x=over_50["age"],
                     y=over_50["chol"],
                     c=over_50["target"],
                     cmap="winter")

ax.set(title= "Heart Disease and Cholesterol levels",
       xlabel="Age",
       ylabel="Cholesterol");

ax.legend(*scatter.legend_elements(), title="Target")
ax.axhline(over_50["chol"].mean(),
           linestyle="--");









#Subplot of chol, age and thalach

fig, (ax0, ax1) = plt.subplots(figsize=(10, 10),
                               nrows= 2,
                               ncols=1,
                               sharex=True)
scatter= ax0.scatter(x=over_50["age"],
                     y=over_50["chol"],
                     c=over_50["target"],
                     cmap="winter")

ax0.set(title= "Heart Disease and Cholesterol levels",
       xlabel="Age",
       ylabel="Cholesterol");
ax0.set_xlim([50, 80])


ax0.legend(*scatter.legend_elements(), title="Target")
ax0.axhline(y=over_50["chol"].mean(),
           linestyle="--")


scatter= ax1.scatter(x=over_50["age"],
                     y=over_50["thalach"],
                     c=over_50["target"],
                     cmap="winter")

ax1.set(title= "Heart Disease and Max Heart Rate",
       xlabel="Age",
       ylabel="Max Heart Rate");
ax1.set_xlim([50, 80])
ax1.set_ylim([60, 200])

ax1.legend(*scatter.legend_elements(), title="Target")
ax1.axhline(y=over_50["thalach"].mean(),
           linestyle="--");


fig.suptitle("Heart Disease Analysis", fontsize=16,
             fontweight="bold")





##Customize our plot
plt.style.available


plt.style.use("seaborn")























