"""
线性回归模型

使用wave数据集
"""

from sklearn.linear_model import LinearRegression

X, y = mglearn.datasets.make_wave(n_samples=60)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

lr = LinearRegression().fit(X_train, y_train)

# 斜率/权重w保存在lr的coef_属性中
print(f"lr.coef_: {lr.coef_}")

# 偏移或截距b保存在lr的intercept_中
print(f"lr.intercept_: {lr.intercept_}")

