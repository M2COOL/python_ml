from sklearn.neighbors import KNeighborsRegressor
X,y = mglearn.datasets.make_wave(n_samples=40)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=5)

reg = KNeighborsRegressor(n_neighbors=3)
reg.fit(X_train, y_train)

print(f"Test set predictions: {reg.predict(X_test)}")