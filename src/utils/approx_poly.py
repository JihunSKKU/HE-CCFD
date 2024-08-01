import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial.chebyshev import Chebyshev
from numpy.polynomial.polynomial import Polynomial

# Define the ReLU function
def relu_func(x):
    return np.where(x > 0, x, 0)

# Define the range and sample more points near the center using Gaussian distribution
np.random.seed(0)
x_center1 = np.random.normal(loc=0, scale=0.05, size=600)  # Centered around 0, more points near the center
x_center2 = np.random.normal(loc=0, scale=0.3, size=300)  # Centered around 0, more points near the center
x_uniform = np.linspace(-1, 1, 100)  # Uniformly distributed points
x = np.concatenate((x_center1, x_center2, x_uniform))
x = np.clip(x, -1, 1)  # Ensure all points are within [-1, 1]

# Sort the x values for better plotting
x = np.sort(x)

# Get the function values at the sample points
y = relu_func(x)

# Define weights for each range

# Fit the Chebyshev polynomial using the weighted least squares method
degree = 4
cheb_fit = Chebyshev.fit(x, y, deg=degree)

# Evaluate the Chebyshev polynomial approximation
y_cheb_fit = cheb_fit(x)

# Print the Chebyshev polynomial coefficients
print("Chebyshev polynomial coefficients (highest degree first):")
print(cheb_fit.coef)

# Convert Chebyshev polynomial to standard polynomial and print the coefficients
poly_fit = cheb_fit.convert(domain=cheb_fit.domain, kind=Polynomial)
print("\nStandard polynomial coefficients (highest degree first):")

scale = 30
print(poly_fit.coef)
print(poly_fit.coef[4] / scale**3)
print(poly_fit.coef[3] / scale**2)
print(poly_fit.coef[2] / scale)
print(poly_fit.coef[1])
print(poly_fit.coef[0] * scale)


# Plot the results
plt.plot(x, y, label='ReLU(x)')
plt.plot(x, y_cheb_fit, label=f'Chebyshev polynomial (degree {degree})')
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.title('ReLU Function and its Polynomial Approximation')
plt.grid(True)
# plt.show()
plt.savefig('./images/relu_approximation.png')  # Save the plot as an image file

# Evaluate the Chebyshev polynomial approximation at specific points
x_points = np.linspace(-1, 1, 6)
y_points = cheb_fit(x_points)

print("\nFunction evaluation at specific points:")
for i in range(len(x_points)):
    print(f"ReLU({x_points[i]}) ≈ {y_points[i]} (approximated) vs. {relu_func(x_points[i])} (actual)")
