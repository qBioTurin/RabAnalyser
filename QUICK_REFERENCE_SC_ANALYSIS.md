# Quick Reference: R Functions for SC Analysis

## One-Minute Overview

Translated Python script → 17 modular R functions across 2 files:
- **clustering_analysis.R** - Core computational functions (8)
- **clustering_visualization.R** - Plotting functions (9)

## Function Quick Guide

### Loading Data
```r
result <- load_and_preprocess_features("data.xlsx")
# Returns: list(data, labels, output_dir)
```

### Feature Selection
```r
feat <- filter_correlated_features(df, threshold=0.7)
# Returns: list(filtered_data, corr_original, corr_filtered)
```

### Dimensionality Reduction
```r
umap <- perform_umap_analysis(df, n_neighbors=20, metric="correlation")
# Returns: list(umap_coords, scaler, umap_model)
```

### Clustering
```r
leiden <- leiden_clustering_optimization(
  umap$umap_model, umap_coords, data_scaled,
  optimal_resolution = 0.004
)
# Returns: list(resolutions, silhouette_scores, dbi_scores, modularity_scores, clusters)
```

### Statistical Analysis
```r
# Proportions
prop <- analyze_subpopulation_proportions(df_with_clusters)

# Feature tests
stats <- statistical_test_clusters(df_with_clusters, correction="fdr")

# Feature importance
importance <- feature_importance_analysis(df_with_clusters, n_trees=100)

# Fingerprints
fingerprints <- subpopulation_fingerprints(df_with_clusters)
```

### Visualization
```r
plot_umap_clusters(umap_coords, clusters)
plot_umap_conditions(umap_coords, conditions)
plot_correlation_matrix(corr_matrix)
plot_leiden_metrics(leiden_result)
visualize_feature_values_umap(umap_coords, feature_data, output_dir=".")
plot_feature_importance(importance_matrix)
plot_subpopulation_proportions(proportions_matrix)
plot_fingerprints(fingerprint_matrix)
plot_statistical_results(stats_result)
```

## Parameter Defaults

| Function | Key Parameter | Default |
|----------|---|---|
| `load_and_preprocess_features()` | `qalpha` | 0.10 |
| `filter_correlated_features()` | `threshold` | 0.7 |
| `perform_umap_analysis()` | `n_neighbors` | 20 |
| | `metric` | "correlation" |
| `leiden_clustering_optimization()` | `n_resolutions` | 50 |
| | `resolution_range` | c(0.0001, 0.1) |
| `statistical_test_clusters()` | `correction` | "fdr" |
| `feature_importance_analysis()` | `n_trees` | 100 |

## Output Interpretation

### Leiden Metrics
- **Silhouette Score**: -1 to +1 (higher = better)
- **Davies-Bouldin Index**: 0 to ∞ (lower = better)
- **Modularity**: 0 to 1 (higher = better)
- **NClusters**: Integer (choose resolution with desired cluster count)

### Statistical Tests
- **p-value**: Unadjusted p-value from Mann-Whitney U test
- **AdjustedPvalue**: After FDR or Bonferroni correction
- **FoldChange**: Median(Cluster1) - Median(Cluster2)
- **Significant**: TRUE if AdjustedPvalue < 0.05

### Feature Importance
- Values from 0 to 1
- Higher = more important for distinguishing clusters
- Heatmap: rows=clusters, columns=features

## Common Workflows

### Exploratory Analysis
```r
# Load data
result <- load_and_preprocess_features("data.xlsx")

# Quick UMAP
umap <- perform_umap_analysis(result$data)
plot_umap_conditions(umap$umap_coords, result$labels)

# Try different resolutions
for (res in c(0.001, 0.004, 0.01)) {
  leiden <- leiden_clustering_optimization(..., optimal_resolution=res)
  cat("Resolution", res, "->", length(unique(leiden$clusters)), "clusters\n")
}
```

### Confirmatory Analysis
```r
# Filtered features
feat <- filter_correlated_features(df)
umap <- perform_umap_analysis(feat$filtered_data)

# Optimize clustering
leiden <- leiden_clustering_optimization(
  umap$umap_model,
  umap$umap_coords,
  scale(feat$filtered_data),
  optimal_resolution = 0.004
)

# Statistical validation
stats <- statistical_test_clusters(df_with_clusters, correction="bonferroni")
```

### Publication Figures
```r
# Main UMAP
p1 <- plot_umap_clusters(umap_coords, clusters, save_path="fig1a.png")

# Statistical results
p2 <- plot_statistical_results(stats$results, save_path="fig1b.png")

# Feature importance
p3 <- plot_feature_importance(importance, save_path="fig1c.png")

# All features on UMAP
visualize_feature_values_umap(umap_coords, features, output_dir=".")
```

## Data Structure Expected

### Input: Excel/CSV File
```
Feature1, Feature2, Feature3, ..., FeatureN, Class
  3.4,     2.1,      1.2,   ...,   0.8,      ConditionA
  2.8,     2.5,      0.9,   ...,   0.6,      ConditionB
  ...
```
- **Columns**: Features (numeric) + Class label (character/factor)
- **Rows**: Cells/observations
- **Last column**: Condition/class labels

### Output: Data Frames

#### UMAP Coordinates
```
  UMAP1  UMAP2
   1.23   0.45
   1.45   0.62
  -0.34  -1.12
  ...
```

#### Clusters
```r
df$Clusters <- c(0, 1, 1, 2, 0, 1, ...)  # Integer cluster IDs
```

#### Statistics Results
```
  Feature  ClusterPair   UStatistic    Pvalue  FoldChange AdjustedPvalue Significant
  Feat1    C0 vs C1      1234.56     0.0023    0.45       0.0069         TRUE
  Feat2    C0 vs C1      2345.67     0.1234    0.12       0.3702         FALSE
  ...
```

## Troubleshooting

| Problem | Diagnosis | Solution |
|---------|-----------|----------|
| UMAP all same point | Check input is numeric | Use `scale()` on features |
| Leiden: 1 cluster | Resolution too low | Increase `resolution_parameter` |
| Too many significant features | Underpowered test | Use stricter correction (Bonferroni) |
| Memory error | Too many features | Pre-filter with `filter_correlated_features()` |
| All feature importances ~0 | Missing values or class imbalance | Check data quality |

## Advanced Options

### Custom UMAP Metric
```r
umap <- perform_umap_analysis(df, metric="euclidean")  # instead of "correlation"
```

### Strict Multiple Testing
```r
stats <- statistical_test_clusters(df, correction="bonferroni")  # stricter than FDR
```

### Manual Resolution Selection
```r
leiden <- leiden_clustering_optimization(
  umap$umap_model, umap_coords, data_scaled
  # No optimal_resolution specified
)
# User inspects plot then selects manually
```

### Save All Plots
```r
plot_leiden_metrics(leiden_result, output_dir=".")  # Saves all metric plots
visualize_feature_values_umap(
  umap_coords, features,
  save_plots=TRUE, output_dir="."
)  # Saves each feature separately
```

## Installation

```r
# For development/local testing:
devtools::load_all()  # from package root

# For use:
library(RabAnalyser)
```

## See Also

- **SC_ANALYSIS_R_TRANSLATION.md** - Full API documentation
- **PYTHON_TO_R_TRANSLATION_SUMMARY.md** - Translation overview

---

**Version**: 1.0  
**Last Updated**: October 2025  
**Package**: RabAnalyser
