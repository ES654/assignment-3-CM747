import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from linearRegression.linearRegression import LinearRegression
from metrics import rmse, mae


N = 30
P = 5
X = pd.DataFrame(np.random.randn(N,P))
y = X[1]

X[5] = X[1]


LR = LinearRegression(True)
LR.fit_vectorised(X,y,10)
y_hat = LR.predict(X)

print('RMSE: ', rmse(y_hat, y))
print('MAE: ', mae(y_hat, y))
print("Values of thetas are:",LR.coef_)