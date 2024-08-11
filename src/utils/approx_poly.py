import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial.chebyshev import Chebyshev
from numpy.polynomial.polynomial import Polynomial

# Define the Swish function
def swish_func(x):
    return x / (1 + np.exp(-x))

# Define the range
x = np.linspace(-5, 5, 1000)  # Uniformly distributed points over [-5, 5]

# Get the function values at the sample points
y = swish_func(x)

# Fit the Chebyshev polynomial using the least squares method
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
print(poly_fit.coef)

# Plot the results
plt.plot(x, y, label='Swish(x)')
plt.plot(x, y_cheb_fit, label=f'Chebyshev polynomial (degree {degree})')
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Swish Function and its Polynomial Approximation')
plt.grid(True)
# plt.show()
plt.savefig('./images/swish_approximation.png')  # Save the plot as an image file

# Evaluate the Chebyshev polynomial approximation at specific points
x_points = np.linspace(-5, 5, 6)
y_points = cheb_fit(x_points)

print("\nFunction evaluation at specific points:")
for i in range(len(x_points)):
    print(f"Swish({x_points[i]}) ≈ {y_points[i]} (approximated) vs. {swish_func(x_points[i])} (actual)")
