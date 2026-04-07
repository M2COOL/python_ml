"""
lasso
"""
from sklearn.linear_model import Lasso
import numpy as np

lasso = Lasso().fit(X_train, y_train)
print(f"Training set score: {lasso.score(X_train, y_train)}")
print(f"Test set score:{lasso.score(X_test, y_test)}")
print(f"Number of features used:{np.sum(lasso.coef_ != 0)}")

'''
默认alpha的lasso在训练集与测试集上表现的都很差，为了降低欠拟合，可以减小alpha，并增加max_iter
'''
lasso001 = Lasso(alpha=0.01, max_iter=10000).fit(X_train, y_train)
print(f"Training set score{lasso001.score(X_train, y_train)}")
print(f"Test set score: {lasso001.score(X_test, y_test)}")
print(f"Number of features used: {lasso001.coef_ != 0}")