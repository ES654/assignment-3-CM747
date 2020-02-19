import numpy as np
import pandas as pd
from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
# Import Autograd modules here
from autograd import grad



class LinearRegression():
    def __init__(self, fit_intercept=True):
        self.fit_intercept = fit_intercept
        self.coef_ = None #Replace with numpy array or pandas series of coefficients learned using using the fit methods
        self.X = None
        self.Y = None
        self.coefs_perepoch = None
        self.errors = None
        pass

    def fit_non_vectorised(self, X1, y1, batch_size, n_iter=100, lr=0.01, lr_type='constant'):
        assert(X1.shape[0]==y1.size)
        assert(lr_type=='constant' or lr_type=='inverse')

        X = X1[:]
        y = y1[:]

        self.X = X
        self.Y = y

        if(self.fit_intercept):
            X.insert(0, "intercept", 1)

        self.coef_ = [0 for i in range(X.shape[1])]
        self.coefs_perepoch = list()
        self.coefs_perepoch.append(self.coef_[:])
        self.errors = list()
        self.errors.append(np.linalg.norm(self.predict(X).values - y.values)**2)

        for epoch in range(1,n_iter+1):
            for batch_start in range(0,X.shape[0],batch_size):
                errors = list()
                for sample in range(batch_start, min(batch_start+batch_size, X.shape[0])):
                    y_hat = 0
                    for feature_index in range(X.shape[1]):
                        y_hat += self.coef_[feature_index]*X.iloc[sample,feature_index]
                    errors.append(y.iat[sample] - y_hat)
                
                for feature_index in range(X.shape[1]):
                    reduction = 0
                    N = 0
                    for sample in range(batch_start, min(batch_start+batch_size, X.shape[0])):
                        reduction += errors[sample]*-1*X.iloc[sample, feature_index]
                        N+=1
                    if(lr_type=='constant'):
                        self.coef_[feature_index] -= lr*2*(reduction/N)
                    else:
                        self.coef_[feature_index] -= (lr/epoch)*2*(reduction/N)
            self.coefs_perepoch.append(self.coef_.copy())
            self.errors.append(np.linalg.norm(self.predict(X).values - y.values)**2)
                    

    def fit_vectorised(self, X1, y1, batch_size, n_iter=100, lr=0.01, lr_type='constant'):
        assert(X1.shape[0]==y1.size)
        assert(lr_type=='constant' or lr_type=='inverse')

        X = X1[:]
        y = y1[:]

        self.X = X
        self.Y = y

        if(self.fit_intercept):
            X.insert(0, "intercept", 1)

        self.coef_ = [0 for i in range(X.shape[1])]
        self.coefs_perepoch = list()
        self.coefs_perepoch.append(self.coef_[:])
        self.errors = list()
        self.errors.append(np.average((y - np.dot(X,self.coef_))**2))

        for epoch in range(1,n_iter+1):
            for batch_start in range(0,X.shape[0],batch_size):
                X_batch = np.array(X.iloc[batch_start: min(batch_start+batch_size, X.shape[0]),:])
                Y_batch = np.array(y[batch_start: min(batch_start+batch_size, X.shape[0])])

                X_batch_T = np.transpose(X_batch)
                X_batch_theta = np.matmul(X_batch, self.coef_)
                reduction = np.matmul(X_batch_T, (X_batch_theta-Y_batch))

                if(lr_type=='constant'):
                    self.coef_ -= lr*(reduction/Y_batch.size)
                else:
                    self.coef_ -= (lr/epoch)*(reduction/Y_batch.size)
            self.coefs_perepoch.append(self.coef_.copy())
            self.errors.append(np.average((y - np.dot(X,self.coef_))**2))


    def fit_autograd(self, X1, y1, batch_size, n_iter=100, lr=0.01, lr_type='constant'):        
        assert(X1.shape[0]==y1.size)
        assert(lr_type=='constant' or lr_type=='inverse')

        X = X1[:]
        y = y1[:]

        self.X = X
        self.Y = y

        if(self.fit_intercept):
            X.insert(0, "intercept", 1)

        def mse(thetas):
            errors = np.dot(X_batch,thetas)-Y_batch
            return np.sum(np.square(errors))/Y_batch.size
        
        gradient = grad(mse)

        self.coef_ = np.array([0.0 for i in range(X.shape[1])])
        self.coefs_perepoch = list()
        self.coefs_perepoch.append(self.coef_[:])
        self.errors = list()
        self.errors.append(np.linalg.norm(self.predict(X).values - y.values)**2)

        for epoch in range(1,n_iter+1):
            for batch_start in range(0,X.shape[0],batch_size):
                X_batch = np.array(X.iloc[batch_start: min(batch_start+batch_size, X.shape[0]),:])
                Y_batch = np.array(y[batch_start: min(batch_start+batch_size, X.shape[0])])

                if(lr_type=='constant'):
                    self.coef_ -= lr*gradient(self.coef_)
                else:
                    self.coef_ -= (lr/epoch)*gradient(self.coef_)
            self.coefs_perepoch.append(self.coef_.copy())
            self.errors.append(np.linalg.norm(self.predict(X).values - y.values)**2)
        


    def fit_normal(self, X1, y1):
        assert(X1.shape[0]==y1.size)

        X = X1[:]
        y = y1[:]

        self.X = X
        self.Y = y

        if(self.fit_intercept):
            X.insert(0, "intercept", 1)
        
        X = np.array(X)
        y = np.array(y)

        XT = np.transpose(X)
        XTX_inv = np.linalg.inv(np.matmul(XT,X))
        K = np.matmul(XT,y)
        self.coef_ = np.matmul(XTX_inv,K)


    def predict(self, X1):
        X = X1[:]
        if(self.fit_intercept and "intercept" not in X.columns):
            X.insert(0, "intercept", 1)
        X = np.array(X)
        y_hat = np.matmul(X,self.coef_)

        return pd.Series(y_hat)


    def plot_surface(self, X1, y1, t_0, t_1):
        assert(X1.shape[0]==y1.size)
        assert(X1.shape[1]==1)
        
        x = np.linspace(-1,1,50)
        y = t_0 + t_1*x

        def cost_func(theta_0, theta_1):
            theta_0 = np.atleast_3d(np.asarray(theta_0))
            theta_1 = np.atleast_3d(np.asarray(theta_1))
            return np.average((y -  (theta_0 + theta_1*x))**2)

        theta0_grid = np.linspace(min(t_0-1,0),t_0+1,101)
        theta1_grid = np.linspace(min(t_1-2,0),t_1+2,101)
        b,m = np.meshgrid(theta0_grid, theta1_grid)
        zs = np.array([cost_func(bp,mp) for bp,mp in zip(np.ravel(b), np.ravel(m))])
        Z = zs.reshape(m.shape)

        self.fit_vectorised(X1,y1,y1.size,50,0.03)

        
        for i in range(len(self.coefs_perepoch)):
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.plot_surface(b, m, Z, rstride=1, cstride=1, cmap=cm.coolwarm, alpha=0.7)
            for j in range(0,i+1):
                k = cost_func(self.coefs_perepoch[j][0],self.coefs_perepoch[j][1])
                ax.scatter([self.coefs_perepoch[j][0]],[self.coefs_perepoch[j][1]],
                            [k], c='r', s=25, marker='.')

            ax.set_xlabel('theta_0')
            ax.set_ylabel('theta_1')
            ax.set_zlabel('Error')
            plt.title("Error:"+str(k))

            plt.savefig("images/plt_surface"+str(i))
            plt.close(fig)




    def plot_line_fit(self, X, y, t_0, t_1):
        """
        Function to plot fit of the line (y vs. X plot) based on chosen value of t_0, t_1. Plot must
        indicate t_0 and t_1 as the title.

        :param X: pd.DataFrame with rows as samples and columns as features (shape: (n_samples, n_features))
        :param y: pd.Series with rows corresponding to output (shape: (n_samples,))
        :param t_0: Value of theta_0 for which to plot the fit
        :param t_1: Value of theta_1 for which to plot the fit

        :return matplotlib figure plotting line fit
        """
        assert(X.shape[0]==y.size)
        assert(X.shape[1]==1)

        self.fit_vectorised(X,y,y.size,50, 0.03)

        x = np.linspace(1,5,100)
        for i in range(len(self.coefs_perepoch)):
            fig = plt.figure()
            plt.scatter(X,y, color='orange')
            
            line = self.coefs_perepoch[i][0] + self.coefs_perepoch[i][1]*x
            plt.plot(x, line, color='blue')

            plt.xlabel('x')
            plt.ylabel('y')
            plt.ylim((1,10))
            plt.xlim((0,6))
            plt.title("t_0=%.2f   t_1=%.2f" % ((self.coefs_perepoch[i][0]),(self.coefs_perepoch[i][1])))
            
            plt.savefig("images/plt_line"+str(i))
            plt.close(fig)
        








    def plot_contour(self, X1, y1, t_0, t_1):
        """
        Plots the RSS as a contour plot. A contour plot is obtained by varying
        theta_0 and theta_1 over a range. Indicates the RSS based on given value of t_0 and t_1, and the
        direction of gradient steps. Uses self.coef_ to calculate RSS.

        :param X: pd.DataFrame with rows as samples and columns as features (shape: (n_samples, n_features))
        :param y: pd.Series with rows corresponding to output (shape: (n_samples,))
        :param t_0: Value of theta_0 for which to plot the fit
        :param t_1: Value of theta_1 for which to plot the fit

        :return matplotlib figure plotting the contour
        """
        assert(X1.shape[0]==y1.size)
        assert(X1.shape[1]==1)
        
        x = np.linspace(-1,1,50)
        y = t_0 + t_1*x

        def cost_func(theta_0, theta_1):
            theta_0 = np.atleast_3d(np.asarray(theta_0))
            theta_1 = np.atleast_3d(np.asarray(theta_1))
            return np.average((y -  (theta_0 + theta_1*x))**2)

        theta0_grid = np.linspace(min(t_0-1,0),t_0+1,101)
        theta1_grid = np.linspace(min(t_1-2,0),t_1+2,101)
        b,m = np.meshgrid(theta0_grid, theta1_grid)
        zs = np.array([cost_func(bp,mp) for bp,mp in zip(np.ravel(b), np.ravel(m))])
        Z = zs.reshape(m.shape)

        self.fit_vectorised(X1,y1,y1.size,50,0.03)

        l = np.array(self.coefs_perepoch)

        for i in range(len(self.coefs_perepoch)):
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.contour(b, m, Z)
            
            ax.scatter(l[:i+1,0],l[:i+1,1], c='r', s=25, marker='.')
            ax.plot(l[:i+1,0],l[:i+1,1], c='r')

            ax.set_xlabel('theta_0')
            ax.set_ylabel('theta_1')

            plt.savefig("images/plt_contour"+str(i))
            plt.close(fig)



        pass
