import numpy as np
import matplotlib.pyplot as plt

# Parameters
sigma_true = 1
mu_true = 0
num_trials = 100
n_values = np.arange(3, 101)  # n = 3,4,...,100

avg_mle_sigma2 = []

# Loop over different sample sizes
for n in n_values:
    mle_estimates = []
    for _ in range(num_trials):
        # Generate n samples from N(0,1)
        sample = np.random.normal(loc=mu_true, scale=sigma_true, size=n)
        # MLE of sigma^2 = (1/n) * sum((x_i - sample_mean)^2)
        sample_mean = np.mean(sample)
        mle_sigma2 = np.sum((sample - sample_mean)**2) / n
        mle_estimates.append(mle_sigma2)
    # Average over trials
    avg_mle_sigma2.append(np.mean(mle_estimates))

# Plot the results
plt.figure(figsize=(8,5))
plt.plot(n_values, avg_mle_sigma2, label='Average MLE of $\sigma^2$')
plt.axhline(y=sigma_true**2, color='r', linestyle='--', label='True $\sigma^2$')
plt.xlabel('Sample size n')
plt.ylabel('Average MLE of $\sigma^2$')
plt.title('Empirical Verification of Bias of MLE of $\sigma^2$')
plt.legend()
plt.grid(True)
plt.show()
