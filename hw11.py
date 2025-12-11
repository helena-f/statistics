import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

# Observed data
observations = [0, 1, 0, 1, 0, 0, 0, 0, 0, 0]
n_successes = sum(observations)  # number of 1s
n_trials = len(observations)     # total observations
n_failures = n_trials - n_successes  # number of 0s

print(f"Data summary:")
print(f"Total observations: {n_trials}")
print(f"Successes (1s): {n_successes}")
print(f"Failures (0s): {n_failures}")
print()

# Define priors (alpha, beta parameters)
priors = [
    (1/2, 1/2, "Beta(1/2, 1/2)"),
    (1, 1, "Beta(1, 1)"),
    (10, 10, "Beta(10, 10)"),
    (100, 100, "Beta(100, 100)")
]

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

# Range of p values
p_values = np.linspace(0, 1, 1000)

# For each prior, calculate and plot the posterior
for idx, (alpha_prior, beta_prior, label) in enumerate(priors):
    # Posterior parameters using Beta-Binomial conjugacy
    # If prior is Beta(α, β) and we observe k successes in n trials,
    # then posterior is Beta(α + k, β + n - k)
    alpha_post = alpha_prior + n_successes
    beta_post = beta_prior + n_failures
    
    # Calculate prior and posterior densities
    prior_pdf = beta.pdf(p_values, alpha_prior, beta_prior)
    posterior_pdf = beta.pdf(p_values, alpha_post, beta_post)
    
    # Plot
    ax = axes[idx]
    ax.plot(p_values, prior_pdf, 'b--', linewidth=2, label='Prior', alpha=0.7)
    ax.plot(p_values, posterior_pdf, 'r-', linewidth=2.5, label='Posterior')
    ax.axvline(x=n_successes/n_trials, color='green', linestyle=':', 
               linewidth=2, label=f'MLE = {n_successes/n_trials:.2f}')
    
    # Calculate posterior mean and mode
    post_mean = alpha_post / (alpha_post + beta_post)
    if alpha_post > 1 and beta_post > 1:
        post_mode = (alpha_post - 1) / (alpha_post + beta_post - 2)
    else:
        post_mode = None
    
    ax.set_xlabel('p', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(f'Prior: {label}\nPosterior: Beta({alpha_post}, {beta_post})', 
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Add text with posterior statistics
    text_str = f'Posterior Mean: {post_mean:.3f}'
    if post_mode is not None:
        text_str += f'\nPosterior Mode: {post_mode:.3f}'
    ax.text(0.98, 0.95, text_str, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    print(f"{label}:")
    print(f"  Posterior: Beta({alpha_post}, {beta_post})")
    print(f"  Posterior Mean: {post_mean:.4f}")
    if post_mode is not None:
        print(f"  Posterior Mode: {post_mode:.4f}")
    print()

plt.show()
# plt.savefig('/mnt/user-data/outputs/bernoulli_posteriors.png', dpi=300, bbox_inches='tight')
print("Plot saved successfully!")
# plt.close()