import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split

data = pd.read_csv("mobile.csv")
sns.pairplot(data)
feature = data.iloc[:, 0:-1].values
label = data.iloc[:, -1].values
X_train, X_test, Y_train, Y_test = train_test_split(feature, label, test_size=0.3, random_state=100)
reg = linear_model.LinearRegression()
reg.fit(X_train, Y_train)
print(reg.coef_)
print(reg.intercept_)
print(reg.predict(X_test))
