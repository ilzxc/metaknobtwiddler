import numpy as np
from sklearn.base import clone
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import LogisticRegression
import seaborn
import pylab as plt
import matplotlib.gridspec as gridspec

data = np.loadtxt('data.csv', delimiter=',', dtype='float32')

X_train, X_test, y_train, y_test = train_test_split(
    data[:,0], data[:,1:], test_size=0.33, random_state=42)
X_train = X_train[:, None]
X_test = X_test[:, None]

# Fit regression model
clfs = {
'DecisionTreeRegressor(max_depth=9)' : DecisionTreeRegressor(max_depth=9),
'DecisionTreeRegressor(max_depth=45)' : DecisionTreeRegressor(max_depth=45),
'RandomForestRegressor(max_depth=9)' : RandomForestRegressor(max_depth=9),
'RandomForestRegressor(max_depth=45)' : RandomForestRegressor(max_depth=45),
'LinearRegression' : LinearRegression(),
'RidgeCV' : RidgeCV() }


fig = plt.figure(figsize=(20, 8))
gs = gridspec.GridSpec(len(clfs), 4)

i = 0
for name, clf in sorted(clfs.items()):
    clf.fit(X_train, y_train)
    y_test_hat = clf.predict(X_test)
    rmse = np.sqrt(np.mean((y_test - y_test_hat)**2))
    ax1 = plt.subplot(gs[i, 0])
    ax2 = plt.subplot(gs[i, 1])
    ax1.set_title(name + " param1")
    ax2.set_title(name + " param2")
    ax1.plot(y_test[:, 0], y_test_hat[:, 0], '.')
    ax2.plot(y_test[:, 1], y_test_hat[:, 1], '.')
    i += 1


i = 0
for name, clf in sorted(clfs.items()):
    m1 = clf
    m2 = clone(clf)

    m1.fit(X_train, y_train[:, 0])
    m2.fit(X_train, y_train[:, 1])

    y_test_hat_1 = m1.predict(X_test)
    y_test_hat_2 = m2.predict(X_test)

    rmse_1 = np.sqrt(np.mean((y_test[:, 0] - y_test_hat_1)**2))
    rmse_2 = np.sqrt(np.mean((y_test[:, 0] - y_test_hat_2)**2))

    ax1 = plt.subplot(gs[i, 2])
    ax2 = plt.subplot(gs[i, 3])
    ax1.set_title(name + " param1")
    ax1.plot(y_test[:, 0], y_test_hat_1, '.')
    ax2.set_title(name + " param2")
    ax2.plot(y_test[:, 1], y_test_hat_2, '.')
    i += 1

plt.tight_layout()
plt.show()