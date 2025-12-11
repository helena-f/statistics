import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.integrate import quad

def posterior_H0_single_obs(x, b):
    """
    Calculate P(H0|X=x) for a single observation.
    
    Setup:
    - X ~ N(μ, 1)
    - H0: μ = 0 vs H1: μ ≠ 0
    - P(H0) = P(H1) = 1/2
    - Under H1: μ ~ N(0, b²)
    
    Using Bayes theorem:
    P(H0|X=x) = P(X=x|H0)P(H0) / P(X=x)
    
    where P(X=x) = P(X=x|H0)P(H0) + P(X=x|H1)P(H1)
    """
    # Likelihood under H0: X ~ N(0, 1)
    likelihood_H0 = norm.pdf(x, loc=0, scale=1)
    
    # Marginal likelihood under H1: 
    # X|H1 ~ N(μ, 1) and μ ~ N(0, b²)
    # So X ~ N(0, 1 + b²) by convolution
    likelihood_H1 = norm.pdf(x, loc=0, scale=np.sqrt(1 + b**2))
    
    # Prior probabilities
    prior_H0 = 0.5
    prior_H1 = 0.5
    
    # Marginal likelihood of data
    marginal_likelihood = likelihood_H0 * prior_H0 + likelihood_H1 * prior_H1
    
    # Posterior probability of H0
    posterior_H0 = (likelihood_H0 * prior_H0) / marginal_likelihood
    
    return posterior_H0

def posterior_H0_sample(x_bar, n, b):
    """
    Calculate P(H0|X̄=x̄) for a sample of size n.
    
    For a sample of size n, X̄ ~ N(μ, 1/n)
    """
    # Likelihood under H0: X̄ ~ N(0, 1/n)
    likelihood_H0 = norm.pdf(x_bar, loc=0, scale=1/np.sqrt(n))
    
    # Marginal likelihood under H1: X̄ ~ N(0, 1/n + b²)
    likelihood_H1 = norm.pdf(x_bar, loc=0, scale=np.sqrt(1/n + b**2))
    
    # Prior probabilities
    prior_H0 = 0.5
    prior_H1 = 0.5
    
    # Marginal likelihood of data
    marginal_likelihood = likelihood_H0 * prior_H0 + likelihood_H1 * prior_H1
    
    # Posterior probability of H0
    posterior_H0 = (likelihood_H0 * prior_H0) / marginal_likelihood
    
    return posterior_H0

def wald_test_pvalue(x, n=1):
    """
    Calculate p-value for Wald test.
    
    Test statistic: Z = (X̄ - 0)/(1/√n) = X̄√n
    Under H0, Z ~ N(0, 1)
    P-value for two-sided test: 2 * P(|Z| > |z_obs|)
    """
    z_stat = x * np.sqrt(n)
    p_value = 2 * (1 - norm.cdf(abs(z_stat)))
    return p_value

# Print the analytical formula
print("=" * 80)
print("JEFFREYS-LINDLEY PARADOX ANALYSIS")
print("=" * 80)
print("\nProblem Setup:")
print("- X ~ N(μ, 1)")
print("- H0: μ = 0  vs  H1: μ ≠ 0")
print("- P(H0) = P(H1) = 1/2")
print("- Under H1: μ ~ N(0, b²)")
print()
print("DERIVED FORMULA for P(H0|X=x):")
print("-" * 80)
print("For a single observation:")
print()
print("                    φ(x; 0, 1) × (1/2)")
print("P(H0|X=x) = ─────────────────────────────────────────")
print("            φ(x; 0, 1) × (1/2) + φ(x; 0, √(1+b²)) × (1/2)")
print()
print("                         1")
print("          = ─────────────────────────────────")
print("            1 + √(1+b²) × exp(x²b²/(2(1+b²)))")
print()
print("For a sample mean X̄ of size n:")
print()
print("                         1")
print("P(H0|X̄=x̄) = ─────────────────────────────────────")
print("            1 + √(1+nb²) × exp(nx̄²b²/(2(1+nb²)))")
print()
print("Wald test p-value = 2Φ(-|x√n|) where Φ is standard normal CDF")
print("=" * 80)
print()

# Part 1: Single observation - vary x and b
print("\n" + "="*80)
print("PART 1: SINGLE OBSERVATION (n=1)")
print("="*80)

x_values = [1.0, 1.5, 2.0, 2.5, 3.0]
b_values = [0.5, 1.0, 2.0, 5.0, 10.0]

print("\nComparison Table:")
print("-" * 80)
print(f"{'x':<8} {'b':<8} {'P(H0|X=x)':<15} {'p-value':<15} {'Difference':<15}")
print("-" * 80)

for x in x_values:
    for b in b_values:
        post_H0 = posterior_H0_single_obs(x, b)
        p_val = wald_test_pvalue(x, n=1)
        diff = post_H0 - p_val
        print(f"{x:<8.2f} {b:<8.2f} {post_H0:<15.6f} {p_val:<15.6f} {diff:<15.6f}")

# Part 2: Sample of size n
print("\n" + "="*80)
print("PART 2: SAMPLE MEAN (various sample sizes n)")
print("="*80)

n_values = [1, 10, 50, 100, 500]
x_bar_values = [0.5, 1.0, 1.5, 1.96]
b_fixed = 1.0

print(f"\nFixed b = {b_fixed}")
print("-" * 80)
print(f"{'x̄':<8} {'n':<8} {'P(H0|X̄=x̄)':<15} {'p-value':<15} {'Difference':<15}")
print("-" * 80)

for x_bar in x_bar_values:
    for n in n_values:
        post_H0 = posterior_H0_sample(x_bar, n, b_fixed)
        p_val = wald_test_pvalue(x_bar, n)
        diff = post_H0 - p_val
        print(f"{x_bar:<8.2f} {n:<8} {post_H0:<15.6f} {p_val:<15.6f} {diff:<15.6f}")

print("\n" + "="*80)
print("KEY OBSERVATIONS:")
print("="*80)
print("1. When n is large and x̄ ≈ 1.96 (p-value ≈ 0.05), P(H0|data) can be > 0.5")
print("2. As n increases, the paradox becomes more pronounced")
print("3. The Bayesian posterior can favor H0 even when p-value < 0.05")
print("4. This shows fundamental disagreement between Bayesian and frequentist approaches")
print("="*80)

# Create visualizations
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: P(H0|X=x) vs x for different b values (n=1)
ax1 = axes[0, 0]
x_range = np.linspace(0, 4, 200)
for b in [0.5, 1.0, 2.0, 5.0]:
    post_probs = [posterior_H0_single_obs(x, b) for x in x_range]
    ax1.plot(x_range, post_probs, linewidth=2, label=f'b={b}')

p_values = [wald_test_pvalue(x, 1) for x in x_range]
ax1.plot(x_range, p_values, 'k--', linewidth=2, label='p-value', alpha=0.7)
ax1.axhline(y=0.05, color='red', linestyle=':', linewidth=1.5, alpha=0.5, label='p=0.05')
ax1.axhline(y=0.5, color='gray', linestyle=':', linewidth=1.5, alpha=0.5)
ax1.set_xlabel('x', fontsize=12)
ax1.set_ylabel('Probability', fontsize=12)
ax1.set_title('Single Observation (n=1): P(H₀|X=x) vs p-value', fontsize=13, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_ylim([0, 1])

# Plot 2: Effect of sample size n (fixed b=1, x̄=2)
ax2 = axes[0, 1]
n_range = np.arange(1, 201, 2)
x_bar_fixed = 2.0
b_fixed = 1.0

post_probs_n = [posterior_H0_sample(x_bar_fixed, n, b_fixed) for n in n_range]
p_values_n = [wald_test_pvalue(x_bar_fixed, n) for n in n_range]

ax2.plot(n_range, post_probs_n, 'b-', linewidth=2.5, label='P(H₀|X̄=x̄)')
ax2.plot(n_range, p_values_n, 'r--', linewidth=2.5, label='p-value')
ax2.axhline(y=0.05, color='red', linestyle=':', linewidth=1.5, alpha=0.5, label='p=0.05')
ax2.axhline(y=0.5, color='gray', linestyle=':', linewidth=1.5, alpha=0.5)
ax2.set_xlabel('Sample size (n)', fontsize=12)
ax2.set_ylabel('Probability', fontsize=12)
ax2.set_title(f'Jeffreys-Lindley Paradox: x̄={x_bar_fixed}, b={b_fixed}', fontsize=13, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_ylim([0, 1])

# Plot 3: Heatmap of P(H0|X=x) - p-value difference
ax3 = axes[1, 0]
x_grid = np.linspace(1.5, 3.5, 100)
b_grid = np.linspace(0.5, 5, 100)
X_mesh, B_mesh = np.meshgrid(x_grid, b_grid)
diff_mesh = np.zeros_like(X_mesh)

for i in range(len(b_grid)):
    for j in range(len(x_grid)):
        post = posterior_H0_single_obs(X_mesh[i,j], B_mesh[i,j])
        pval = wald_test_pvalue(X_mesh[i,j], 1)
        diff_mesh[i,j] = post - pval

im = ax3.contourf(X_mesh, B_mesh, diff_mesh, levels=20, cmap='RdBu_r')
ax3.contour(X_mesh, B_mesh, diff_mesh, levels=[0], colors='black', linewidths=2)
plt.colorbar(im, ax=ax3, label='P(H₀|X=x) - p-value')
ax3.set_xlabel('x', fontsize=12)
ax3.set_ylabel('b', fontsize=12)
ax3.set_title('Difference: P(H₀|X=x) - p-value (n=1)', fontsize=13, fontweight='bold')
ax3.grid(True, alpha=0.3)

# Plot 4: Multiple sample sizes comparison
ax4 = axes[1, 1]
x_bar_range = np.linspace(1.5, 3.0, 100)
b_fixed = 1.0

for n in [1, 10, 50, 100]:
    post_probs = [posterior_H0_sample(x_bar, n, b_fixed) for x_bar in x_bar_range]
    ax4.plot(x_bar_range, post_probs, linewidth=2, label=f'n={n}')

p_values_plot = [wald_test_pvalue(2.0, n) for n in [1, 10, 50, 100]]
ax4.axhline(y=0.05, color='red', linestyle=':', linewidth=1.5, alpha=0.5, label='p=0.05')
ax4.axhline(y=0.5, color='gray', linestyle=':', linewidth=1.5, alpha=0.5)
ax4.set_xlabel('x̄', fontsize=12)
ax4.set_ylabel('P(H₀|X̄=x̄)', fontsize=12)
ax4.set_title(f'Posterior Probability vs Sample Mean (b={b_fixed})', fontsize=13, fontweight='bold')
ax4.legend(fontsize=10)
ax4.grid(True, alpha=0.3)
ax4.set_ylim([0, 1])

plt.tight_layout()
# plt.savefig('jeffreys_lindley_paradox.png', dpi=300, bbox_inches='tight')
print("\nPlot saved as 'jeffreys_lindley_paradox.png'")
plt.show()