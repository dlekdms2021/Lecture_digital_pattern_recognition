
#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import make_pipeline

# Simulate dataset
np.random.seed(0)
x = np.linspace(-2, 2, 100)
y = 1 + 2 * x - 0.5 * x**3 + np.random.normal(0, 1, size=x.shape)
df_daily = pd.DataFrame({"Temperature": x, "Load": y})

# Prepare input and output
X = df_daily[["Temperature"]]
y = df_daily["Load"]
x_test = np.linspace(X.min()[0], X.max()[0], 200).reshape(-1, 1)

# Plot polynomial regression with different degrees and regularization
fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=False)
degrees = [3, 60]
alphas = [0.0, 1.0]  # 0.0 = no regularization, 1.0 = ridge

for i, deg in enumerate(degrees):
    for j, alpha in enumerate(alphas):
        if alpha == 0.0:
            model = make_pipeline(PolynomialFeatures(degree=deg), LinearRegression())
            label = f"Poly deg={deg} (No Reg)"
        else:
            model = make_pipeline(PolynomialFeatures(degree=deg), Ridge(alpha=alpha))
            label = f"Poly deg={deg} + Ridge(alpha={alpha})"

        model.fit(X, y)
        y_pred = model.predict(x_test)

        ax = axes[i, j]
        ax.scatter(X, y, color='gray', s=20, label="Data")
        ax.plot(x_test, y_pred, color='blue', label=label)
        ax.set_title(label)
        ax.grid(True)
        ax.legend()

plt.tight_layout()
plt.show()
