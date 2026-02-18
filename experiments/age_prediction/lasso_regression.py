from sklearn.model_selection import train_test_split
from sklearn.linear_model import  Lasso
from sklearn.preprocessing import MinMaxScaler
from scipy import stats

from utilz.Dataset import load_dataset
from utilz.preprocessing_utilz import *
from utilz.helpers import *


meta_path = r"../../../data/samples_pancreatic.xlsx"
data_path = r"../../../data/counts_pancreatic.csv"

ds = load_dataset(data_path, meta_path, label_col="Group")

y = ds.age

y = y.dropna().astype(int)
# Prediction of age is only meaningful for healthy samples
healthy_idx = ds.y[ds.y == HEALTHY].index
idx = y.index.intersection(healthy_idx)

X = ds.X.loc[idx]
y = y.loc[idx]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=42)

#model = LinearRegression(fit_intercept=True)
model = Lasso(alpha = 0.11, max_iter=10000)

pipeline = Pipeline([
    ('NoneInformativeGeneReductor', NoneInformativeGeneReductor()),
    ('scaler', MinMaxScaler()),
    ('model', model)
])

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

print("Mean Absolute Error:", np.mean(np.abs(y_test - y_pred)))
print("Root Mean Squared Error:", np.sqrt(np.mean((y_test - y_pred) ** 2)))
print("R^2 Score:", pipeline.score(X_test, y_test))

plt.figure()
plt.scatter(y_test, y_pred, alpha=0.4)
min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
plt.axis('equal')
plt.xlabel("True Age")
plt.ylabel("Predicted Age")
plt.title("Linear Regression Age Prediction")
plt.legend()
plt.grid(alpha=0.3)
plt.show()

residuals = y_test - y_pred
stats.probplot(residuals, dist="norm", plot=plt)
plt.title("QQ plot residuals")
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(np.sort(pipeline["model"].coef_.ravel()))
plt.xlabel("Feature Index")
plt.ylabel("Coefficient Value")
plt.title("Linear Regression Coefficients")
plt.show()


coef = pipeline.named_steps["model"].coef_.ravel()
n_zero = (coef == 0).sum()
n_total = coef.size

print(f"Zerowe współczynniki: {n_zero}/{n_total} ({n_zero/n_total:.2%})")
