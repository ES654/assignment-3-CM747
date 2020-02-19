import numpy as np
import matplotlib.pyplot as plt
from preprocessing.polynomial_features import PolynomialFeatures
from linearRegression.linearRegression import LinearRegression
import pandas as pd

x = np.array([i*np.pi/180 for i in range(60,300,4)])
np.random.seed(10)  #Setting seed for reproducibility
y = 4*x + 7 + np.random.normal(0,3,len(x))


weights = list()
degree = [i for i in range(1,10)]
for d in degree:
    poly = PolynomialFeatures(d, False)
    X = poly.transform(np.transpose([x]))
    LR = LinearRegression(True)
    LR.fit_normal(pd.DataFrame(data=X), pd.Series(y))
    weights.append(np.linalg.norm(LR.coef_))

plt.figure(figsize=(8,6))
plt.bar(x=degree, height=weights)
plt.yscale("log")
plt.xlabel("Degree")
plt.ylabel("Magnitude of theta vector")
plt.title("Linear Regression using Normal Equation")

plt.show()


