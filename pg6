import numpy as np
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess

np.random.seed(42)
X = np.linspace(0, 2 * np.pi, 100)
y = np.sin(X) + 0.1 * np.random.randn(100)

lowess_result = lowess(y, X, frac=0.3) 

plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='red', label='Training Data', alpha=0.7)
plt.plot(lowess_result[:, 0], lowess_result[:, 1], color='blue', label='LOWESS␣Fit', linewidth=2)
plt.xlabel('X', fontsize=12)
plt.ylabel('y', fontsize=12)
plt.title('Locally Weighted Regression using statsmodels', fontsize=14)
plt.legend(fontsize=10)
plt.grid(alpha=0.3)
plt.show()
