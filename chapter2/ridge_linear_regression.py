"""
岭回归

"""

from sklearn.linear_model import Ridge

ridge = Ridge().fit(X_train, y_train)
print(f"Training set score: {ridge.score(X_train, y_train)}")

print(f"Test set score: {ridge.score(X_test, y_test)}")

"""
使用alpha调整简单性与训练集性能
"""

ridge10 = Ridge(alpha=10).fit(X_train, y_train)

print(f"Training set score{ridge10.score(X_train, y_train)}")
print(f"Test set score{ridge10.score(X_test, y_test)}")


