import numpy as np
import matplotlib.pyplot as plt

class Linreg:
    def __init__(self,learning_rate=0.001, stop_value=10**(-10),alpha = 1):
        self.learning_rate = learning_rate
        self.stop_value = stop_value
        self.alpha = alpha
    def fit(self,X,y):
        n,n_features = X.shape
        #y = (y-np.mean(y))/np.var(y)
        #k = np.var(X,axis=0)
        self.mean = np.mean(X,axis=0)
        self.std = np.std(X,axis=0)
        X = (X-np.mean(X,axis=0)) / np.std(X,axis=0)

        self.bias, self.weights = 0, np.zeros(n_features)
        pred_db,pred_dw = 0, np.zeros(n_features)

        while True:
            y_pred = X @ self.weights + self.bias

            db = 1/n * np.sum(y_pred - y)
            dw = 1/n * X.T @ (y_pred-y) + 0.01/n * self.weights
            self.bias -= self.learning_rate*db
            self.weights -= self.learning_rate*dw

            abs_diff_db = np.abs(pred_db - db)
            abs_diff_dw = np.abs(pred_dw - dw)

            if abs_diff_db<self.stop_value:
                if abs_diff_dw.all()<self.stop_value:
                    break
            pred_db = db
            pred_dw = dw
        self.bias -= np.sum(self.mean*self.weights/self.std)
        self.weights /= self.std

    def predict(self,X):
        #X = (X - self.mean) / self.std
        return X@ self.weights + self.bias


lr = Linreg()

X = np.linspace(-2,10,200)
y = 12 * np.sin(X) + 3 + np.random.normal(0,5,*X.shape) * 1

x = np.column_stack((X,np.sin(X)))
#print(X.reshape((-1,1)))
lr.fit(x,y)

plt.plot(X,y)
plt.plot(X, x@lr.weights+lr.bias)
plt.show()

print(lr.weights, lr.bias)
