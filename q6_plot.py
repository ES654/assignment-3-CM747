import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from preprocessing.polynomial_features import PolynomialFeatures
from linearRegression.linearRegression import LinearRegression
import seaborn as sns

np.random.seed(10)  #Setting seed for reproducibility


# plt.figure(figsize=(8,6))

plot_details = {'N':list(), 'D':list(), 'T':list()}

N_range = [i for i in range(200,800,50)]
degree = [1,3,5,7,9]

for N in N_range:
    x = np.array([i*np.pi/180 for i in range(60,60+4*N,4)])
    y = 4*x + 7 + np.random.normal(0,3,len(x))
    # weights = list()
    for d in degree:
        poly = PolynomialFeatures(d, False)
        X = poly.transform(np.transpose([x]))
        LR = LinearRegression(True)
        LR.fit_normal(pd.DataFrame(data=X), pd.Series(y))
        plot_details['N'].append(N)
        plot_details['D'].append(d)
        plot_details['T'].append(np.linalg.norm(LR.coef_))
        # weights.append(np.linalg.norm(LR.coef_))
    # plt.plot(degree, weights, label="N="+str(N))

labels = np.array(plot_details['T'])
labels = labels.reshape((len(N_range),len(degree)))

df = pd.DataFrame(data=plot_details)
heatmap1_data = pd.pivot_table(df, values='T', index=['N'], columns='D')
sns.heatmap(heatmap1_data, cmap="YlOrRd", annot=labels)
plt.title("Magnitude of theta with varying degree and number of training samples")
plt.show()

# plt.yscale("log")
# plt.xlabel("Degree")
# plt.ylabel("Magnitude of theta vector")
# plt.legend()
# plt.title("Linear Regression using Normal Equation")

# plt.show()