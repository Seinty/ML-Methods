import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from mlxtend.plotting import plot_decision_regions


def decision_boundary_plot(X, y, X_train, y_train, clf, feature_indexes, title=None):
    feature1_name, feature2_name = X.columns[feature_indexes]
    X_feature_columns = X.values[:, feature_indexes]
    X_train_feature_columns = X_train[:, feature_indexes]
    clf.fit(X_train_feature_columns, y_train)

    plot_decision_regions(X=X_feature_columns, y=y.values, clf=clf)
    plt.xlabel(feature1_name)
    plt.ylabel(feature2_name)
    plt.title(title)
    plt.show()


class GaussianNaive:
    def fit(self, X,y):
        classes, cls_count = np.unique(y,return_counts = True)
        n_classes = len(classes)
        self.priors = cls_count/len(y)

        self.X_cls_mean = np.array([np.mean(X[y==c],axis = 0) for c in range(n_classes)])
        self.X_stds = np.array([np.std(X[y==c],axis = 0) for c in range(n_classes)])

    def pdf(self,x,mean,std):
        return 1/(np.sqrt(2*np.pi)*std)*np.exp(-((x-mean)/std)**2)

    def predict(self,X):
        pdfs = np.array([self.pdf(x,self.X_cls_mean,self.X_stds) for x in X])
        posteriors = self.priors * np.prod(pdfs,axis = 2)

        return np.argmax(posteriors, axis = 1)

X1, y1 = load_iris(return_X_y=True, as_frame=True)

X1.iloc[-1] = [5.4,4.5,4.5,0.5]
y1.iloc[-1] = 0
X1_train, X1_test, y1_train, y1_test = train_test_split(X1.values, y1.values, random_state=3,test_size=0.2)
#print(X1, y1, sep='\n')

nb_clf = GaussianNaive()
nb_clf.fit(X1_train, y1_train)
nb_clf_pred_res = nb_clf.predict(X1_test)
nb_clf_accuracy = accuracy_score(y1_test, nb_clf_pred_res)

print(f'Naive Bayes classifier accucacy: {nb_clf_accuracy}')
#print(nb_clf_pred_res)

#print()
feature_indexes = [2, 3]
title1 = 'GaussianNB surface'
decision_boundary_plot(X1, y1, X1_train, y1_train, nb_clf, feature_indexes, title1)