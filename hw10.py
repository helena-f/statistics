import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Set random seed for reproducibility
np.random.seed(42)

# Parameters
mu_true = 5
sigma = 1
n = 100

# ============================================================================
# Part (a): Simulate a data set with n=100 observations
# ============================================================================
X = np.random.normal(mu_true, sigma, n)
print(f"Part (a): Simulated {n} observations")
print(f"Sample mean: {np.mean(X):.4f}")
print(f"Sample std: {np.std(X, ddof=1):.4f}\n")

# ============================================================================
# Part (b): Find the posterior density with f(mu) = 1 (uniform prior)
# ============================================================================
# Prior: f(mu) = 1 (uniform, improper prior)
# Likelihood: X_i ~ Normal(mu, 1)
# Posterior: mu | X ~ Normal(X_bar, 1/n)

x_bar = np.mean(X)
posterior_mean = x_bar
posterior_variance = sigma**2 / n
posterior_std = np.sqrt(posterior_variance)

print(f"Part (b): Posterior density")
print(f"Posterior: mu | X ~ Normal({posterior_mean:.4f}, {posterior_variance:.4f})")
print(f"Posterior mean: {posterior_mean:.4f}")
print(f"Posterior std: {posterior_std:.4f}\n")

# Plot the posterior density
mu_range = np.linspace(posterior_mean - 4*posterior_std, 
                       posterior_mean + 4*posterior_std, 1000)
posterior_density = stats.norm.pdf(mu_range, posterior_mean, posterior_std)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(mu_range, posterior_density, 'b-', linewidth=2, label='Posterior density')
plt.axvline(posterior_mean, color='r', linestyle='--', 
            label=f'Posterior mean = {posterior_mean:.4f}')
plt.axvline(mu_true, color='g', linestyle='--', 
            label=f'True μ = {mu_true}')
plt.xlabel('μ')
plt.ylabel('Density')
plt.title('Part (b): Posterior Density of μ')
plt.legend()
plt.grid(True, alpha=0.3)

# ============================================================================
# Part (c): Simulate 1,000 draws from the posterior and compare
# ============================================================================
n_draws = 1000
posterior_samples = np.random.normal(posterior_mean, posterior_std, n_draws)

print(f"Part (c): Simulated {n_draws} draws from posterior")
print(f"Sample mean of draws: {np.mean(posterior_samples):.4f}")
print(f"Sample std of draws: {np.std(posterior_samples):.4f}")

# Plot histogram and compare with theoretical density
plt.subplot(1, 2, 2)
plt.hist(posterior_samples, bins=50, density=True, alpha=0.6, 
         color='skyblue', edgecolor='black', label='Histogram of samples')
plt.plot(mu_range, posterior_density, 'b-', linewidth=2, 
         label='Theoretical posterior')
plt.axvline(posterior_mean, color='r', linestyle='--', 
            label=f'Posterior mean = {posterior_mean:.4f}')
plt.axvline(mu_true, color='g', linestyle='--', 
            label=f'True μ = {mu_true}')
plt.xlabel('μ')
plt.ylabel('Density')
plt.title('Part (c): Histogram vs Theoretical Posterior')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('posterior_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# Summary Statistics
# ============================================================================
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"True μ: {mu_true}")
print(f"Sample mean (X_bar): {x_bar:.4f}")
print(f"Posterior mean: {posterior_mean:.4f}")
print(f"Posterior std: {posterior_std:.4f}")
print(f"95% Credible Interval: [{posterior_mean - 1.96*posterior_std:.4f}, "
      f"{posterior_mean + 1.96*posterior_std:.4f}]")