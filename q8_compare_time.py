import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from linearRegression.linearRegression import LinearRegression
import time

N_range = [i for i in range(30,500)]
P_range = [i for i in range(100,1000)]

fit_timesVsN_grad = list()
fit_timesVsN_norm = list()
P = 1000
for N in N_range:
    X = pd.DataFrame(np.random.randn(N, P))
    y = pd.Series(np.random.randn(N))
    
    LR = LinearRegression(fit_intercept=False)
    startTime = time.time()
    LR.fit_normal(X,y)
    endTime = time.time()
    fit_timesVsN_norm.append((endTime-startTime)*1000000000000)

    LR = LinearRegression(fit_intercept=False)
    startTime = time.time()
    LR.fit_vectorised(X,y,y.size)
    endTime = time.time()
    fit_timesVsN_grad.append((endTime-startTime)*1000000000000)
    print(N)

fit_timesVsP_norm = list()
fit_timesVsP_grad = list()
N = 100
for P in P_range:
    X = pd.DataFrame(np.random.randn(N, P))
    y = pd.Series(np.random.randn(N))
    
    LR = LinearRegression(fit_intercept=False)
    startTime = time.time()
    LR.fit_normal(X,y)
    endTime = time.time()
    fit_timesVsP_norm.append((endTime-startTime)*10)

    LR = LinearRegression(fit_intercept=False)
    startTime = time.time()
    LR.fit_vectorised(X,y,y.size)
    endTime = time.time()
    fit_timesVsP_grad.append((endTime-startTime)*10)
    print(P)


fig = plt.figure(figsize=(18,5))


plt.subplot(1,2,1)
plt.plot(N_range, fit_timesVsN_norm, color='orange', label='Normal Eq')
plt.plot(N_range, fit_timesVsN_grad, color='blue', label='Gradient Descent')
plt.xlabel("N (No. of Samples)")
plt.ylabel("Emperical_fit_time")
plt.legend()

plt.subplot(1,2,2)
plt.plot(P_range, fit_timesVsP_norm, color='orange', label='Normal Eq')
plt.plot(P_range, fit_timesVsP_grad, color='blue', label='Gradient Descent')
plt.xlabel("P (No. of Attributes/Features)")
plt.ylabel("Emperical_fit_time")
plt.legend()

plt.show()