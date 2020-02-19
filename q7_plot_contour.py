import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from linearRegression.linearRegression import LinearRegression
import pandas as pd

x = np.array([i*np.pi/180 for i in range(60,300,4)])
np.random.seed(10)  #Setting seed for reproducibility
y = 1*x + 2 + np.random.normal(0,1,len(x))

LR = LinearRegression(True)

LR.plot_surface(pd.DataFrame(data=x), pd.Series(y), 2, 1)

LR.plot_line_fit(pd.DataFrame(data=x), pd.Series(y),2,1)

LR.plot_contour(pd.DataFrame(data=x), pd.Series(y),2,1)