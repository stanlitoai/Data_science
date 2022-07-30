import sklearn
import pandas as pd
import numpy as np

df = pd.read_csv("heart-disease.csv")

df

X = df.drop("target", axis=1)

y = df["target"]

# Choose the right model and hyperparameters

from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators = 60)

clf.get_params()



#Slipt out data
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#fitting our model

clf.fit(X_train, y_train)

#Make some predictions
y_preds = clf.predict(X_test)

y_preds


#Evaluate the model on the training data and test data

clf.score(X_train, y_train)


clf.score(X_test, y_test)

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


print(classification_report(y_test, y_preds))


confusion_matrix(y_test, y_preds)

accuracy_score(y_test, y_preds)


##Improve the model

np.random.seed(42)
for i in range(10, 100, 10):
    print(f"Trying model with {i} estimators....")
    clf = RandomForestClassifier(n_estimators=i).fit(X_train, y_train)
    print(f"Model accuracy on test set: {clf.score(X_test, y_test) * 100: .2f}%")
    print("")




# Save the model
import pickle 

pickle.dump(clf, open("random_forest_model_2.pkl", "wb"))

loaded_model = pickle.load(open("random_forest_model_2.pkl", "rb"))

loaded_model.score(X_test, y_test)






car_sales = pd.read_csv("car-sales-extended.csv")


car_sales.head()



X = car_sales.drop("Price", axis=1)

y = car_sales["Price"]

X.head()
y.head()

###SPlit into train_test

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)



###Build a machine learning model

from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor()

model.fit(X_train, y_train)

model.score(X_train, y_train)

model.score(X_test, y_test)








#USing ONEHOT ENCODER(turn the categories into numbers)
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

categorical_features = ["Make", "Colour", "Doors"]

one_hot = OneHotEncoder()
transformer = ColumnTransformer([("one_hot",
                                  one_hot,
                                  categorical_features)],
                                remainder="passthrough")

transformed_X = transformer.fit_transform(X)

transformed_X




####################USing dummies
dummies = pd.get_dummies(car_sales[["Make", "Colour", "Doors"]])

dummies.head()



###SPlit into train_test
np.random.seed(42)
X_train, X_test, y_train, y_test = train_test_split(transformed_X, y, test_size=0.2)



###Build a machine learning model

from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor()

model.fit(X_train, y_train)

model.score(X_train, y_train)

model.score(X_test, y_test)




##USing Dummies

###SPlit into train_test
np.random.seed(42)
X_train, X_test, y_train, y_test = train_test_split(dummies, y, test_size=0.2)



###Build a machine learning model

from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor()

model.fit(X_train, y_train)

model.score(X_train, y_train)

model.score(X_test, y_test)



####Working with missing datas

car_sales_missing = pd.read_csv("car-sales-extended-missing-data.csv")

car_sales_missing.head()

car_sales_missing.isna().sum()




X = car_sales_missing.drop("Price", axis=1)

y = car_sales_missing["Price"]

X.isna().sum()
y.head()

###SPlit into train_test

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)





#USing ONEHOT ENCODER(turn the categories into numbers)
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

categorical_features = ["Make", "Colour", "Doors"]

one_hot = OneHotEncoder()
transformer = ColumnTransformer([("one_hot",
                                  one_hot,
                                  categorical_features)],
                                remainder="passthrough")

transformed_X = transformer.fit_transform(X)

transformed_X




######Option1: Fill missing data with pandas

car_sales_missing["Make"].fillna("missing", inplace=True)

car_sales_missing["Colour"].fillna("missing", inplace=True)

car_sales_missing["Odometer (KM)"].fillna(car_sales_missing["Odometer (KM)"].mean(), inplace=True)

car_sales_missing["Doors"].fillna(4, inplace=True)



car_sales_missing.isna().sum()


###############OR

car_sales_missing.dropna(inplace=True)






X = car_sales_missing.drop("Price", axis=1)

y = car_sales_missing["Price"]

X.isna().sum()
y.head()

###SPlit into train_test

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)





#USing ONEHOT ENCODER(turn the categories into numbers)
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

categorical_features = ["Make", "Colour", "Doors"]

one_hot = OneHotEncoder()
transformer = ColumnTransformer([("one_hot",
                                  one_hot,
                                  categorical_features)],
                                remainder="passthrough")

transformed_X = transformer.fit_transform(car_sales_missing)

transformed_X
























