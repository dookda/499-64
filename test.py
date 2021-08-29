
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


test = r"E:\StudyLAB\lab_programing\pyqgis\test_499\test.csv"
trian = r"E:\StudyLAB\lab_programing\pyqgis\test_499\train.csv"

data_train = pd.read_csv(trian)
data_test = pd.read_csv(test)

# print("See the data sample:\n")
# print(data_train.head())
# print("See the summary of features:\n")
# print(data_train.info())
# print("Calculate basic stat of each feature:\n")
# print(data_train.describe())

plt.figure(figsize=(12, 9))
plt.scatter(data_train["x"], data_train["y"])
plt.xlabel("x")
plt.ylabel("y")
# plt.show()

X_train = np.array(data_train["x"]).reshape(-1, 1)
y_train = np.array(data_train["y"]).reshape(-1, 1)
X_test = np.array(data_test["x"]).reshape(-1, 1)
y_test = np.array(data_test["y"]).reshape(-1, 1)

print("X_train shape = " + str(X_train.shape))
print("y_train shape = " + str(y_train.shape))
print("X_test shape = " + str(X_test.shape))
print("y_test shape = " + str(y_test.shape))

np.any(np.isnan(X_train))
np.all(np.isfinite(X_train))

np.any(np.isnan(y_train))
np.all(np.isfinite(y_train))

lr = LinearRegression().fit(X_train, y_train)
print("LR coefficient is", lr.coef_)
print("LR intercept is", lr.intercept_)
