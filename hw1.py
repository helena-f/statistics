# For n = 10, 20, 30, . . . , 10000, sample n i.i.d. samples from N (0, 1) i.e. the random variable X with
# density fX (x) = 1√2π exp(−x2/2). Let  ̄xn be the corresponding sample average. Plot  ̄xn as a function
# of n. Describe the behavior as n increases. What does the Law of Large Numbers suggest will happen
# as n → ∞?

import numpy as np
import matplotlib.pyplot as plt

n_values = np.arange(10, 10000, 10) # n = 10, 20, 30, . . . , 10000
normal_sample_means = []
cauchy_sample_means = []

# Generate data for Normal distribution
for n in n_values:
    # density fX (x) = 1√2π exp(−x2/2) depicts normal distribution
    samples = np.random.normal(0, 1, n) # i.i.d. samples from N (0, 1)
    normal_sample_means.append(np.mean(samples))

# Generate data for Cauchy distribution
for n in n_values:
    samples = np.random.standard_cauchy(n)
    cauchy_sample_means.append(np.mean(samples))

# Plot Normal distribution
plt.figure(1)
plt.plot(n_values, normal_sample_means)
plt.xlabel('n')
plt.ylabel('Sample Mean of Normal Distribution')
plt.title('Normal Sample Mean vs. n')
plt.legend(['Sample Mean'])
plt.grid(True)
plt.show()

# Plot Cauchy distribution
plt.figure(2)
plt.plot(n_values, cauchy_sample_means)
plt.xlabel('n')
plt.ylabel('Sample Mean of Cauchy Distribution')
plt.title('Cauchy Sample Mean vs. n')
plt.legend(['Sample Mean'])
plt.grid(True)
plt.show()

