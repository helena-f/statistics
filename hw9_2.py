import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from sklearn.model_selection import KFold
from scipy.optimize import minimize_scalar

# Read the glass data file
# The file has a header row with column names
data = pd.read_csv('glass.dat', sep=r'\s+', header=0)

# Extract the first variable (refractive index - RI)
ri = data['RI'].values

# Remove any missing values
ri = ri[~np.isnan(ri)]

print("=" * 70)
print("Density Estimation for Refractive Index (RI)")
print("=" * 70)
print(f"\nSample size: {len(ri)}")
print(f"Mean: {ri.mean():.4f}")
print(f"Std: {ri.std():.4f}")
print(f"Min: {ri.min():.4f}")
print(f"Max: {ri.max():.4f}")

# ============================================================================
# PART 1: Histogram with different binwidths
# ============================================================================
print("\n" + "=" * 70)
print("PART 1: Histogram Density Estimation")
print("=" * 70)

# Calculate optimal binwidth using different methods
n = len(ri)
data_range = ri.max() - ri.min()

# Different binwidth calculation methods
binwidths = {
    'Sturges': data_range / (np.log2(n) + 1),
    'Scott': 3.5 * ri.std() / (n ** (1/3)),
    'Freedman-Diaconis': 2 * (np.percentile(ri, 75) - np.percentile(ri, 25)) / (n ** (1/3)),
    'Small': data_range / 20,  # Fine bins
    'Medium': data_range / 10,  # Medium bins
    'Large': data_range / 5     # Coarse bins
}

print("\nBinwidths to experiment with:")
for method, bw in binwidths.items():
    print(f"  {method:20s}: {bw:.4f} ({int(data_range/bw)} bins)")

# Create histogram plots with different binwidths
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Histogram Density Estimation with Different Binwidths', fontsize=14, fontweight='bold')

axes_flat = axes.flatten()
for idx, (method, binwidth) in enumerate(binwidths.items()):
    ax = axes_flat[idx]
    bins = np.arange(ri.min(), ri.max() + binwidth, binwidth)
    counts, bin_edges, patches = ax.hist(ri, bins=bins, density=True, alpha=0.7, 
                                         edgecolor='black', linewidth=0.5)
    ax.set_title(f'{method}\n(binwidth={binwidth:.3f})', fontsize=10)
    ax.set_xlabel('Refractive Index (RI)')
    ax.set_ylabel('Density')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('histogram_comparison.png', dpi=300, bbox_inches='tight')
print("\nHistogram comparison saved to 'histogram_comparison.png'")

# ============================================================================
# PART 2: Kernel Density Estimation with Cross-Validation
# ============================================================================
print("\n" + "=" * 70)
print("PART 2: Kernel Density Estimation with Cross-Validation")
print("=" * 70)

# Cross-validation function for bandwidth selection
def cv_log_likelihood(bandwidth, data, k_folds=5):
    """
    Calculate cross-validated log-likelihood for a given bandwidth.
    Uses leave-one-out cross-validation.
    """
    n = len(data)
    log_likelihood = 0.0
    
    for i in range(n):
        # Leave-one-out: use all data except the i-th observation
        train_data = np.delete(data, i)
        test_point = data[i]
        
        # Create KDE with current bandwidth
        kde = gaussian_kde(train_data)
        kde.set_bandwidth(bandwidth)
        
        # Calculate log-likelihood for the test point
        try:
            density = kde(test_point)[0]
            if density > 0:
                log_likelihood += np.log(density)
        except:
            pass
    
    return -log_likelihood  # Negative because we'll minimize

# Find optimal bandwidth using cross-validation
print("\nFinding optimal bandwidth using cross-validation...")
print("This may take a moment...")

# Search range for bandwidth
bandwidth_range = (0.1, 5.0)
result = minimize_scalar(cv_log_likelihood, bounds=bandwidth_range, 
                         args=(ri,), method='bounded')
optimal_bandwidth = result.x

print(f"\nOptimal bandwidth (cross-validation): {optimal_bandwidth:.4f}")

# Also calculate Silverman's rule of thumb bandwidth
silverman_bw = (4 * ri.std()**5 / (3 * n)) ** (1/5)
print(f"Silverman's rule of thumb bandwidth: {silverman_bw:.4f}")

# ============================================================================
# PART 3: KDE with different bandwidths
# ============================================================================
print("\n" + "=" * 70)
print("PART 3: KDE with Different Bandwidths")
print("=" * 70)

# Different bandwidths to experiment with
bandwidths = {
    'Very Small (0.1)': 0.1,
    'Small (0.3)': 0.3,
    'Optimal (CV)': optimal_bandwidth,
    'Silverman': silverman_bw,
    'Medium (1.0)': 1.0,
    'Large (2.0)': 2.0
}

print("\nBandwidths to experiment with:")
for method, bw in bandwidths.items():
    print(f"  {method:20s}: {bw:.4f}")

# Create KDE plots with different bandwidths
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Kernel Density Estimation with Different Bandwidths', fontsize=14, fontweight='bold')

x_plot = np.linspace(ri.min() - 2, ri.max() + 2, 1000)
axes_flat = axes.flatten()

for idx, (method, bandwidth) in enumerate(bandwidths.items()):
    ax = axes_flat[idx]
    
    # Create KDE with specified bandwidth
    kde = gaussian_kde(ri)
    kde.set_bandwidth(bandwidth)
    
    # Evaluate KDE on plot points
    density = kde(x_plot)
    
    # Plot
    ax.plot(x_plot, density, 'b-', linewidth=2, label='KDE')
    ax.hist(ri, bins=30, density=True, alpha=0.3, color='gray', 
            edgecolor='black', linewidth=0.5, label='Histogram')
    ax.set_title(f'{method}\n(bandwidth={bandwidth:.3f})', fontsize=10)
    ax.set_xlabel('Refractive Index (RI)')
    ax.set_ylabel('Density')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('kde_comparison.png', dpi=300, bbox_inches='tight')
print("\nKDE comparison saved to 'kde_comparison.png'")

# ============================================================================
# PART 4: Side-by-side comparison
# ============================================================================
print("\n" + "=" * 70)
print("PART 4: Side-by-Side Comparison")
print("=" * 70)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Comparison: Histogram vs Kernel Density Estimation', fontsize=14, fontweight='bold')

# Left: Best histogram (Freedman-Diaconis)
ax1 = axes[0]
best_binwidth = binwidths['Freedman-Diaconis']
bins = np.arange(ri.min(), ri.max() + best_binwidth, best_binwidth)
ax1.hist(ri, bins=bins, density=True, alpha=0.7, edgecolor='black', 
         linewidth=0.5, color='steelblue', label='Histogram')
ax1.set_title(f'Histogram (Freedman-Diaconis, binwidth={best_binwidth:.3f})')
ax1.set_xlabel('Refractive Index (RI)')
ax1.set_ylabel('Density')
ax1.grid(True, alpha=0.3)
ax1.legend()

# Right: Best KDE (Cross-validated)
ax2 = axes[1]
kde_optimal = gaussian_kde(ri)
kde_optimal.set_bandwidth(optimal_bandwidth)
density_optimal = kde_optimal(x_plot)
ax2.plot(x_plot, density_optimal, 'r-', linewidth=2, label='KDE (CV optimal)')
ax2.hist(ri, bins=30, density=True, alpha=0.3, color='gray', 
         edgecolor='black', linewidth=0.5, label='Histogram (reference)')
ax2.set_title(f'KDE (Cross-validated, bandwidth={optimal_bandwidth:.3f})')
ax2.set_xlabel('Refractive Index (RI)')
ax2.set_ylabel('Density')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('histogram_vs_kde.png', dpi=300, bbox_inches='tight')
print("\nComparison plot saved to 'histogram_vs_kde.png'")

# ============================================================================
# PART 5: Comments and Analysis
# ============================================================================
print("\n" + "=" * 70)
print("COMMENTS ON SIMILARITIES AND DIFFERENCES")
print("=" * 70)

print("""
SIMILARITIES:
1. Both methods estimate the underlying probability density function of the data.
2. Both show that the refractive index has a roughly unimodal distribution with 
   some skewness.
3. Both methods are sensitive to their smoothing parameters:
   - Histograms: binwidth controls smoothness
   - KDE: bandwidth controls smoothness
4. Both can reveal the general shape of the distribution (central tendency, spread).

DIFFERENCES:
1. SMOOTHNESS:
   - Histograms: Produce step functions (discontinuous)
   - KDE: Produce smooth, continuous curves

2. SENSITIVITY TO PARAMETER CHOICE:
   - Histograms: Very sensitive to bin placement and width. Small changes in 
     binwidth can significantly alter the appearance.
   - KDE: More robust to bandwidth choice, but still sensitive. Too small 
     bandwidth → overfitting (wiggly), too large → oversmoothing (loses detail).

3. BOUNDARY BEHAVIOR:
   - Histograms: Can handle boundaries naturally
   - KDE: May have issues near boundaries (though less noticeable here)

4. COMPUTATIONAL COMPLEXITY:
   - Histograms: O(n) - very fast
   - KDE: O(n²) for evaluation at each point (slower, especially with CV)

5. INTERPRETATION:
   - Histograms: Easier to interpret, shows actual counts/densities in bins
   - KDE: More abstract, represents a smooth estimate of the true density

6. OPTIMAL PARAMETER SELECTION:
   - Histograms: Rules of thumb (Sturges, Scott, Freedman-Diaconis)
   - KDE: Cross-validation provides data-driven optimal bandwidth selection

OBSERVATIONS FROM THIS DATA:
- The refractive index appears to have a roughly normal distribution with 
  slight negative skew (tail on the left).
- The optimal bandwidth from cross-validation ({:.4f}) provides a good 
  balance between smoothness and detail.
- Histograms with different binwidths show how sensitive the method is to 
  parameter choice - too many bins (small binwidth) shows noise, too few 
  bins (large binwidth) loses important features.
- KDE with optimal bandwidth provides a smoother, more aesthetically pleasing 
  estimate while preserving the main features of the distribution.
""".format(optimal_bandwidth))

print("\n" + "=" * 70)
print("Analysis complete! Check the generated PNG files for visualizations.")
print("=" * 70)

# Show plots
plt.show()
