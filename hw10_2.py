import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import gaussian_kde

# Set random seed for reproducibility
np.random.seed(42)

def analyze_density_estimation(X, true_density, x_range, title, sample_size):
    """
    Analyze histogram and KDE for a given sample
    
    Parameters:
    - X: sample data
    - true_density: function that computes true density
    - x_range: range for plotting
    - title: plot title
    - sample_size: n value
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'{title} (n = {sample_size})', fontsize=16, fontweight='bold')
    
    # ========================================================================
    # Part (i): Histograms with different bin sizes
    # ========================================================================
    
    # Compute different bin size choices
    # Sturges' rule: k = ceil(log2(n) + 1)
    sturges_bins = int(np.ceil(np.log2(len(X)) + 1))
    
    # Scott's rule: h = 3.5 * std(X) / n^(1/3)
    scott_h = 3.5 * np.std(X) / (len(X) ** (1/3))
    scott_bins = int(np.ceil((x_range[1] - x_range[0]) / scott_h))
    
    # Freedman-Diaconis rule: h = 2 * IQR(X) / n^(1/3)
    iqr = np.percentile(X, 75) - np.percentile(X, 25)
    fd_h = 2 * iqr / (len(X) ** (1/3))
    fd_bins = int(np.ceil((x_range[1] - x_range[0]) / fd_h))
    
    bin_choices = [
        (10, 'Few bins (10)', 0),
        (sturges_bins, f'Sturges ({sturges_bins})', 1),
        (scott_bins, f'Scott ({scott_bins})', 2)
    ]
    
    x_plot = np.linspace(x_range[0], x_range[1], 1000)
    true_pdf = true_density(x_plot)
    
    for bins, label, idx in bin_choices:
        ax = axes[0, idx]
        ax.hist(X, bins=bins, density=True, alpha=0.6, 
                color='skyblue', edgecolor='black', label=f'Histogram')
        ax.plot(x_plot, true_pdf, 'r-', linewidth=2, label='True density')
        ax.set_xlabel('x')
        ax.set_ylabel('Density')
        ax.set_title(label)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # ========================================================================
    # Part (ii): Kernel Density Estimation with different bandwidths
    # ========================================================================
    
    # Silverman's rule of thumb: h = 0.9 * min(std, IQR/1.34) * n^(-1/5)
    silverman_h = 0.9 * min(np.std(X), iqr/1.34) * (len(X) ** (-1/5))
    
    # Scott's rule for KDE: h = std(X) * n^(-1/5)
    scott_kde_h = np.std(X) * (len(X) ** (-1/5))
    
    bandwidth_choices = [
        (silverman_h * 0.3, f'Small h = {silverman_h * 0.3:.4f}', 0),
        (silverman_h, f'Silverman h = {silverman_h:.4f}', 1),
        (silverman_h * 3, f'Large h = {silverman_h * 3:.4f}', 2)
    ]
    
    for bw, label, idx in bandwidth_choices:
        ax = axes[1, idx]
        kde = gaussian_kde(X, bw_method=bw/np.std(X))
        kde_values = kde(x_plot)
        
        ax.plot(x_plot, kde_values, 'b-', linewidth=2, label='KDE')
        ax.plot(x_plot, true_pdf, 'r-', linewidth=2, label='True density')
        ax.fill_between(x_plot, kde_values, alpha=0.3)
        ax.set_xlabel('x')
        ax.set_ylabel('Density')
        ax.set_title(label)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig, sturges_bins, scott_bins, fd_bins, silverman_h, scott_kde_h

# ============================================================================
# Case (a): X ~ Unif([0,1]), n = 100
# ============================================================================
print("="*70)
print("Case (a): X ~ Unif([0,1]), n = 100")
print("="*70)

n_a = 100
X_a = np.random.uniform(0, 1, n_a)
true_density_a = lambda x: np.where((x >= 0) & (x <= 1), 1.0, 0.0)

fig_a, sturges_a, scott_a, fd_a, silverman_a, scott_kde_a = analyze_density_estimation(
    X_a, true_density_a, [-0.2, 1.2], 'Uniform([0,1])', n_a
)
plt.savefig('case_a_uniform_100.png', dpi=300, bbox_inches='tight')

print(f"Sturges bins: {sturges_a}")
print(f"Scott bins: {scott_a}")
print(f"FD bins: {fd_a}")
print(f"Silverman bandwidth: {silverman_a:.4f}")
print(f"Scott KDE bandwidth: {scott_kde_a:.4f}\n")

# ============================================================================
# Case (b): X ~ Unif([0,1]), n = 10000
# ============================================================================
print("="*70)
print("Case (b): X ~ Unif([0,1]), n = 10000")
print("="*70)

n_b = 10000
X_b = np.random.uniform(0, 1, n_b)

fig_b, sturges_b, scott_b, fd_b, silverman_b, scott_kde_b = analyze_density_estimation(
    X_b, true_density_a, [-0.2, 1.2], 'Uniform([0,1])', n_b
)
plt.savefig('case_b_uniform_10000.png', dpi=300, bbox_inches='tight')

print(f"Sturges bins: {sturges_b}")
print(f"Scott bins: {scott_b}")
print(f"FD bins: {fd_b}")
print(f"Silverman bandwidth: {silverman_b:.4f}")
print(f"Scott KDE bandwidth: {scott_kde_b:.4f}\n")

# ============================================================================
# Case (c): X ~ N(0,1), n = 100
# ============================================================================
print("="*70)
print("Case (c): X ~ N(0,1), n = 100")
print("="*70)

n_c = 100
X_c = np.random.normal(0, 1, n_c)
true_density_c = lambda x: stats.norm.pdf(x, 0, 1)

fig_c, sturges_c, scott_c, fd_c, silverman_c, scott_kde_c = analyze_density_estimation(
    X_c, true_density_c, [-4, 4], 'Normal(0,1)', n_c
)
plt.savefig('case_c_normal_100.png', dpi=300, bbox_inches='tight')

print(f"Sturges bins: {sturges_c}")
print(f"Scott bins: {scott_c}")
print(f"FD bins: {fd_c}")
print(f"Silverman bandwidth: {silverman_c:.4f}")
print(f"Scott KDE bandwidth: {scott_kde_c:.4f}\n")

# ============================================================================
# Case (d): X ~ N(0,1), n = 10000
# ============================================================================
print("="*70)
print("Case (d): X ~ N(0,1), n = 10000")
print("="*70)

n_d = 10000
X_d = np.random.normal(0, 1, n_d)

fig_d, sturges_d, scott_d, fd_d, silverman_d, scott_kde_d = analyze_density_estimation(
    X_d, true_density_c, [-4, 4], 'Normal(0,1)', n_d
)
plt.savefig('case_d_normal_10000.png', dpi=300, bbox_inches='tight')

print(f"Sturges bins: {sturges_d}")
print(f"Scott bins: {scott_d}")
print(f"FD bins: {fd_d}")
print(f"Silverman bandwidth: {silverman_d:.4f}")
print(f"Scott KDE bandwidth: {scott_kde_d:.4f}\n")

# ============================================================================
# Summary and Analysis
# ============================================================================
print("\n" + "="*70)
print("SUMMARY AND QUALITATIVE ANALYSIS")
print("="*70)

print("\n1. HISTOGRAM BEHAVIOR:")
print("-" * 70)
print("Bias-Variance Tradeoff:")
print("  • Few bins (high bias, low variance): Oversmooths, misses features")
print("  • Many bins (low bias, high variance): Noisy, overfits to sample")
print("  • Optimal bins balance bias and variance")
print("\nEffect of sample size:")
print("  • Small n (100): Need fewer bins to avoid empty bins")
print("  • Large n (10000): Can use more bins for finer detail")

print("\n2. KDE BANDWIDTH BEHAVIOR:")
print("-" * 70)
print("Bias-Variance Tradeoff:")
print("  • Small h (low bias, high variance): Wiggly, captures noise")
print("  • Large h (high bias, low variance): Oversmooths, misses features")
print("  • Optimal h balances bias and variance")
print("\nEffect of sample size:")
print("  • Optimal bandwidth h ∝ n^(-1/5)")
print("  • As n increases, we can use smaller h to capture more detail")

print("\n3. DISTRIBUTION-SPECIFIC OBSERVATIONS:")
print("-" * 70)
print("Uniform distribution:")
print("  • Histograms better capture sharp boundaries")
print("  • KDE tends to oversmooth at boundaries (boundary bias)")
print("\nNormal distribution:")
print("  • Both methods work well")
print("  • KDE provides smoother estimates")
print("  • Silverman's rule works particularly well for Normal data")

print("\n4. THEORETICAL CONNECTIONS:")
print("-" * 70)
print("Optimal bin width: h* ∝ n^(-1/3)")
print("Optimal KDE bandwidth: h* ∝ n^(-1/5)")
print("\nMISE (Mean Integrated Squared Error):")
print("  • Histogram: MISE ∝ n^(-2/3)")
print("  • KDE: MISE ∝ n^(-4/5)")
print("  → KDE converges faster asymptotically!")

plt.show()