
import pandas as pd
import matplotlib.pyplot as plt


data = pd.read_csv("pzzz.csv")
X = data.iloc[:, 0]
Y = data.iloc[:, 1]
plt.scatter(X, Y)
plt.show()

m = 0
c = 0
L = 0.0001
n = float(len(X))
epochs = 1000
for i in range(epochs):
    y = m*X + c
    D_m = sum(X*(y - Y))/n
    D_c = sum(y - Y)/n
    m = m - L*D_m
    c = c - L*D_c

print(m, c)

Y_pred = m*X + c
plt.scatter(X, Y)
plt.plot([min(X), max(X)], [min(Y_pred), max(Y_pred)], color='red')
plt.show()
x = int(input(""))
print(m*x + c)
