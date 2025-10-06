import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

T = 100
eps = np.random.normal(0, 1, T)

phi_values = [0.5, 0.9, 1.0, 1.1]  # different AR(1) coefficients
plt.figure(figsize=(12, 6))

for phi in phi_values:
    X = np.zeros(T)
    for t in range(1, T):
        X[t] = phi * X[t-1] + eps[t]
    plt.plot(X, label=f'AR(1) φ={phi}')

plt.title("AR(1) Time Series with Different φ₁ Values")
plt.xlabel("Time")
plt.ylabel("Value")
plt.legend()
plt.show()
