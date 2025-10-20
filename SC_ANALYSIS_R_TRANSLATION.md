# R Translation of SC_MultipleCNDTN_Analysis_V3.py

## Overview

The Python script `SC_MultipleCNDTN_Analysis_V3.py` performs comprehensive single-cell subpopulation analysis. This document describes the R translation with modular functions for each analysis step.

## Python Script Analysis Workflow

The original Python script performs the following workflow:

```
1. File Loading & Preprocessing
   ├── Load Excel file with feature matrix
   ├── Apply soft-threshold denoising
   ├── Subsample to equal group sizes
   └── Output: Preprocessed data frame

2. Feature Selection & Correlation Analysis
   ├── Filter correlated features (Pearson r > 0.7)
   ├── Generate correlation matrices (before/after)
   └── Output: Reduced feature set + correlation heatmaps

3. UMAP Dimensionality Reduction
   ├── Standardize features (z-score normalization)
   ├── Compute 2D UMAP projection
   ├── Plot UMAP colored by condition
   └── Output: UMAP coordinates

4. Leiden Clustering Optimization
   ├── Test multiple resolution parameters (0.0001 to 0.1)
   ├── Compute quality metrics for each resolution:
   │   ├── Silhouette Score (higher better, -1 to 1)
   │   ├── Davies-Bouldin Index (lower better, 0 to ∞)
   │   └── Modularity Score (higher better, 0 to 1)
   ├── Plot metrics vs. resolution
   ├── Select optimal resolution
   └── Output: Clusters + metric plots

5. Cluster Characterization
   ├── Subpopulation proportions analysis
   │   ├── Calculate cell percentages per cluster/condition
   │   ├── Chi-square test for global differences
   │   ├── Pairwise post-hoc tests (Z-test or Fisher exact)
   │   └── FDR correction
   ├── Create stacked bar plots
   └── Output: Proportions table + statistical results

6. Statistical Feature Analysis
   ├── Mann-Whitney U tests per feature per cluster
   ├── Fold change calculation (median difference)
   ├── Multiple comparison correction (FDR)
   ├── Create dot plots with size/color encoding
   └── Export to Excel with results per feature

7. Feature Visualization on UMAP
   ├── For each feature:
   │   ├── Create UMAP colored by feature value
   │   ├── Use blue-white-green colormap (-0.4 to 0.4)
   │   └── Save PNG and SVG
   └── Output: Feature value heatmaps

8. Feature Importance Analysis
   ├── Random Forest classification
   ├── For 2 clusters: Binary classification
   ├── For >2 clusters: One-vs-rest per cluster
   ├── Extract feature importances
   └── Output: Importance heatmap

9. Subpopulation Fingerprints
   ├── Compute mean feature value per cluster
   ├── Create heatmap with custom colors
   │   ├── Blue gradient (left, negative values)
   │   ├── White band (center, near-zero)
   │   └── Green gradient (right, positive values)
   └── Output: Fingerprint heatmap
```

## R Module Structure

The translation is organized into modular R functions, each handling one analysis step:

### Core Analysis Functions (clustering_analysis.R)

#### 1. `load_and_preprocess_features(data_path, qalpha=0.10, gamma=0.05)`
**Purpose**: Load and preprocess feature data

**Input**: 
- Path to Excel/CSV file (last column = condition labels)

**Parameters**:
- `qalpha`: Soft-threshold noise floor
- `gamma`: Soft-threshold steepness

**Output**: 
```r
list(
  data = preprocessed_df,
  labels = condition_labels,
  output_dir = base_directory
)
```

**Details**:
- Applies soft-threshold formula: `sign(x) * (|x| - qalpha) * tanh((|x| - qalpha)/gamma)`
- Subsamples all groups to equal minimum count
- Uses dplyr for efficient grouping

---

#### 2. `filter_correlated_features(df, threshold=0.7)`
**Purpose**: Remove redundant correlated features

**Input**:
- Data frame with numeric features (last column = labels)

**Parameters**:
- `threshold`: Correlation threshold for filtering (default 0.7)

**Output**:
```r
list(
  filtered_data = reduced_features_df,
  corr_original = original_correlation_matrix,
  corr_filtered = filtered_correlation_matrix
)
```

**Details**:
- Greedy approach: keeps first feature, removes later correlated ones
- Returns both original and filtered correlation matrices
- Useful for visualizing feature redundancy

---

#### 3. `perform_umap_analysis(df, n_neighbors=20, min_dist=1, metric="correlation", seed=42)`
**Purpose**: Compute UMAP projection for visualization

**Input**:
- Numeric feature data frame

**Parameters**:
- `n_neighbors`: Number of neighbors (default 20)
- `min_dist`: Minimum distance in UMAP (default 1)
- `metric`: Distance metric (default "correlation")
- `seed`: Random seed for reproducibility

**Output**:
```r
list(
  umap_coords = UMAP_coordinates_df,
  scaler = scaling_info,
  umap_model = fitted_UMAP_model
)
```

---

#### 4. `leiden_clustering_optimization(umap_model, umap_coords, data_scaled, ...)`
**Purpose**: Find optimal Leiden clustering resolution

**Parameters**:
- `resolution_range`: Range of resolutions to test (default c(0.0001, 0.1))
- `n_resolutions`: Number of resolution values (default 50)
- `optimal_resolution`: Which resolution to use for final clustering
- `seed`: Random seed

**Output**:
```r
list(
  resolutions = resolution_values,
  silhouette_scores = silhouette_scores,
  dbi_scores = davies_bouldin_scores,
  modularity_scores = modularity_scores,
  num_clusters = cluster_counts,
  clusters = final_cluster_assignments,  # if optimal_resolution specified
  partition = igraph_partition_object
)
```

**Metrics Interpretation**:
- **Silhouette**: -1 (worst) to +1 (best). High is better.
- **DBI**: 0 (best) to ∞. Low is better.
- **Modularity**: 0 to 1. High is better.

---

#### 5. `analyze_subpopulation_proportions(df_features, clusters_col="Clusters", condition_col="Condition")`
**Purpose**: Analyze cell proportions and test for condition effects

**Input**:
- Data frame with cluster and condition columns

**Output**:
```r
list(
  proportions = proportions_df,  # clusters x conditions matrix
  chi2_pvalue = global_test_pvalue,
  pairwise_tests = pairwise_results_df  # if significant
)
```

**Statistical Tests**:
- Global: Chi-square test
- Pairwise: Z-test (if expected ≥ 5) or Fisher exact test
- Multiple comparison correction: FDR

---

#### 6. `statistical_test_clusters(df_features, clusters_col="Clusters", correction="fdr")`
**Purpose**: Mann-Whitney U tests between clusters for each feature

**Parameters**:
- `correction`: "fdr" or "bonferroni"

**Output**:
```r
list(
  results = full_results_df,  # All features & clusters
  significant_features = sig_features_df  # Only p < 0.05
)
```

**Result Columns**:
- Feature: Feature name
- ClusterPair: Which clusters compared
- UStatistic: Mann-Whitney U statistic
- Pvalue: Unadjusted p-value
- FoldChange: Median(Cluster1) - Median(Cluster2)
- AdjustedPvalue: After correction
- Significant: TRUE if p < 0.05 after correction

---

#### 7. `feature_importance_analysis(df_features, clusters_col="Clusters", n_trees=100, seed=42)`
**Purpose**: Random Forest classification importance analysis

**Output**:
```r
list(
  importance_matrix = importance_heatmap_data,  # Clusters x Features
  model_stats = list(
    n_clusters = number_of_clusters,
    n_features = number_of_features,
    n_samples = number_of_samples
  )
)
```

**Classification Strategy**:
- 2 clusters: Binary classification
- >2 clusters: One-vs-rest for each cluster

---

#### 8. `subpopulation_fingerprints(df_features, clusters_col="Clusters")`
**Purpose**: Compute mean feature profile per subpopulation

**Output**:
```r
list(
  fingerprint_matrix = mean_features_per_cluster,  # Clusters x Features
  summary_stats = list(
    n_clusters = nrow,
    n_features = ncol,
    mean_values = column_means
  )
)
```

### Visualization Functions (clustering_visualization.R)

#### Plot Functions

1. **`plot_umap_clusters()`** - UMAP colored by cluster assignment
2. **`plot_umap_conditions()`** - UMAP colored by condition
3. **`plot_correlation_matrix()`** - Heatmap of feature correlations
4. **`plot_leiden_metrics()`** - Resolution vs. quality metrics
5. **`visualize_feature_values_umap()`** - UMAP for each feature
6. **`plot_feature_importance()`** - Random Forest importance heatmap
7. **`plot_subpopulation_proportions()`** - Stacked bar chart by condition
8. **`plot_fingerprints()`** - Mean feature values per cluster
9. **`plot_statistical_results()`** - Dot plot of test results

All plot functions:
- Return ggplot objects for further customization
- Support saving to PNG and SVG
- Use publication-quality styling

## Usage Example

```r
library(RabAnalyser)

# Step 1: Load and preprocess
result <- load_and_preprocess_features("data/features.xlsx")
df <- result$data
output_dir <- result$output_dir

# Step 2: Feature selection
feat_result <- filter_correlated_features(df, threshold = 0.7)
df_filtered <- feat_result$filtered_data
plot_correlation_matrix(feat_result$corr_original, "Original Features")
plot_correlation_matrix(feat_result$corr_filtered, "Filtered Features")

# Step 3: UMAP
umap_result <- perform_umap_analysis(df_filtered)
umap_coords <- umap_result$umap_coords
plot_umap_conditions(umap_coords, df$Class)

# Step 4: Leiden clustering optimization
leiden_result <- leiden_clustering_optimization(
  umap_result$umap_model,
  umap_coords,
  scale(df_filtered),
  optimal_resolution = 0.004
)
plot_leiden_metrics(leiden_result, output_dir)

# Step 5: Add clusters to features
df_filtered$Clusters <- leiden_result$clusters
df_filtered$Condition <- df$Class
plot_umap_clusters(umap_coords, leiden_result$clusters)

# Step 6: Proportions analysis
prop_result <- analyze_subpopulation_proportions(df_filtered)
plot_subpopulation_proportions(prop_result$proportions)

# Step 7: Statistical tests
stat_result <- statistical_test_clusters(df_filtered)
plot_statistical_results(stat_result$results)

# Step 8: Feature visualization
visualize_feature_values_umap(
  umap_coords,
  df_filtered[, -which(names(df_filtered) %in% c("Clusters", "Condition"))],
  output_dir = output_dir
)

# Step 9: Feature importance
importance_result <- feature_importance_analysis(df_filtered)
plot_feature_importance(importance_result$importance_matrix)

# Step 10: Fingerprints
fingerprint_result <- subpopulation_fingerprints(df_filtered)
plot_fingerprints(fingerprint_result$fingerprint_matrix)
```

## Key Differences from Python

| Aspect | Python | R |
|--------|--------|---|
| **Data Import** | pandas.read_excel() | readxl::read_excel() |
| **Soft-threshold** | numpy array operations | apply() with vectorization |
| **UMAP** | umap library | umap R package |
| **Clustering** | igraph + leidenalg | leiden R package + igraph |
| **Statistics** | scipy.stats | R base stats functions |
| **Multiple Testing** | statsmodels | stats::p.adjust() |
| **Visualization** | matplotlib + seaborn | ggplot2 + pheatmap |
| **Random Forest** | sklearn | randomForest package |

## Dependencies

Install required packages:

```r
install.packages(c(
  "dplyr", "tidyr", "ggplot2", "pheatmap",
  "readxl", "umap", "leiden", "igraph",
  "cluster", "clusterCrit", "randomForest"
))
```

## Output Files

When `save_plots=TRUE`, functions generate:

```
output_dir/
├── CorrMatrix_original.png/svg
├── CorrMatrix_filtered.png/svg
├── UMAP_clusters.png/svg
├── UMAP_by_condition.png/svg
├── UMAP_visual_[feature1].png/svg
├── UMAP_visual_[feature2].png/svg
├── Leiden_Silhouette_DBI.png
├── Leiden_Modularity.png
├── Leiden_NClusters.png
├── Feature_Importance.png/svg
├── Cell_Proportions.png/svg
├── Subpopulation_Fingerprints.png/svg
└── Statistical_Results.png/svg
```

## Parameter Recommendations

### For UMAP
- `n_neighbors`: 15-30 (higher captures global structure)
- `min_dist`: 0.1-1.0 (lower = more clustered)
- `metric`: "correlation" for standardized features, "euclidean" for raw

### For Leiden
- `resolution_range`: c(0.0001, 0.5) for exploratory
- `resolution_range`: c(0.001, 0.01) for fine-tuning around optimal value

### For Feature Selection
- `threshold`: 0.7 (typical for correlation-based filtering)
- Can adjust up to 0.9 for more aggressive filtering

### For Statistical Tests
- Use "fdr" correction for exploratory analysis
- Use "bonferroni" for confirmatory studies

## Troubleshooting

**Q: UMAP coordinates all identical?**
A: Check that input features are standardized (z-score). Use `scale()`.

**Q: Leiden clustering producing 1 cluster?**
A: Increase `resolution_parameter`. Try 0.01-0.1 range.

**Q: Too many significant features in stats?**
A: Use stricter p-value threshold or more conservative correction (Bonferroni vs FDR).

**Q: Feature importance values all zero?**
A: Ensure Random Forest is fitting correctly. Check for missing values.

## References

- UMAP: McInnes L. et al. (2018). Uniform Manifold Approximation and Projection
- Leiden: Traag VA. et al. (2019). From Louvain to Leiden clustering
- Mann-Whitney U test: Non-parametric test for comparing two groups

