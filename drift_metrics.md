## Understanding Drift Metrics

### Population Stability Index (PSI)

**What it measures:** How much the distribution of a variable has shifted between two datasets.

**How it works:** Divides data into bins (typically 10), calculates the percentage of observations in each bin for both reference and current data, then measures the divergence.

**Formula:** `PSI = Σ (Current% - Reference%) × ln(Current% / Reference%)`

| PSI Value | Interpretation | Action |
|-----------|----------------|--------|
| < 0.1 | No significant shift | Continue monitoring |
| 0.1 - 0.2 | Moderate shift | Investigate cause |
| 0.2 - 0.5 | Significant shift | Plan model update |
| > 0.5 | Severe shift | Immediate retraining |

### Kolmogorov-Smirnov Test (KS-test)

**What it measures:** The maximum difference between cumulative distribution functions of two samples.

**How it works:** Computes the empirical CDF for both samples and finds the point of maximum divergence.

| KS Statistic | p-value | Interpretation |
|--------------|---------|----------------|
| Low (< 0.1) | > 0.05 | Distributions are similar |
| High (> 0.2) | < 0.05 | Distributions are different |
| Very High (> 0.3) | < 0.01 | Distributions are very different |

### Jensen-Shannon Divergence

**What it measures:** A symmetric measure of similarity between two probability distributions.

**How it works:** Computes the average of KL divergence in both directions. Bounded between 0 (identical) and 1 (completely different).

| JS Value | Interpretation |
|----------|----------------|
| < 0.05 | Very similar distributions |
| 0.05 - 0.15 | Some divergence, monitor |
| 0.15 - 0.30 | Significant divergence |
| > 0.30 | Major distribution shift |

### Wasserstein Distance (Earth Mover's Distance)

**What it measures:** The minimum "cost" of transforming one distribution into another, measured in units of the variable.

**How it works:** Conceptually, measures how much probability mass needs to be moved and how far.

**Interpretation:** A Wasserstein distance of 5 for temperature means the distributions are separated by about 5°C on average. The interpretation is scale-dependent.

### Summary Table

| Metric | Range | Best For | Threshold Guidance |
|--------|-------|----------|-------------------|
| PSI | 0 to ∞ | Overall drift magnitude | < 0.1 stable, > 0.2 action |
| KS Statistic | 0 to 1 | Distribution shape changes | p-value < 0.05 significant |
| JS Divergence | 0 to 1 | Bounded comparison | < 0.1 stable, > 0.2 action |
| Wasserstein | 0 to ∞ | Interpretable distance | Domain-specific |
| Chi-Square | 0 to ∞ | Categorical variables | p-value < 0.05 significant |