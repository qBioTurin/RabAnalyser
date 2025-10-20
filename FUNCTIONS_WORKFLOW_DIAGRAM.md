# RabAnalyser Translation: Function Execution Flowchart

## Visual Workflow (ASCII Flowchart)

```
┌─────────────────────────────────────────────────────────────────────┐
│                    SC_MultipleCNDTN_Analysis_V3.py                  │
│                         Translated to R                             │
│                      17 Modular Functions                           │
└─────────────────────────────────────────────────────────────────────┘

PHASE 1: DATA LOADING & PREPROCESSING
═════════════════════════════════════════════════════════════════════
                               ↓
                    [1] load_and_preprocess_features()
                     ├─ Load Excel/CSV
                     ├─ Apply soft-threshold
                     └─ Subsample to equal sizes
                               ↓

PHASE 2: FEATURE SELECTION
═════════════════════════════════════════════════════════════════════
                               ↓
                   [2] filter_correlated_features()
                    ├─ Greedy correlation filtering
                    ├─ Generate correlation matrices
                    └─ Return reduced feature set
                               ↓
                    [3] plot_correlation_matrix() ◄─ VISUALIZATION
                    ├─ Original features heatmap
                    └─ Filtered features heatmap
                               ↓

PHASE 3: DIMENSIONALITY REDUCTION
═════════════════════════════════════════════════════════════════════
                               ↓
                    [4] perform_umap_analysis()
                     ├─ Standardize features
                     ├─ Compute UMAP projection
                     └─ Return 2D coordinates
                               ↓
                   [5] plot_umap_conditions() ◄─ VISUALIZATION
                    └─ Color by experimental condition
                               ↓

PHASE 4: CLUSTERING OPTIMIZATION
═════════════════════════════════════════════════════════════════════
                               ↓
                [6] leiden_clustering_optimization()
                 ├─ Test multiple resolutions
                 ├─ Compute silhouette scores
                 ├─ Compute DBI scores
                 ├─ Compute modularity scores
                 └─ Return optimal clusters
                               ↓
                  ┌────────────┴────────────┐
                  ↓                         ↓
        [7] plot_leiden_metrics()   [8] plot_umap_clusters()
            ◄─ VISUALIZATION           ◄─ VISUALIZATION
            Metrics vs. resolution     Color by cluster
                               ↓
                          (SELECT RESOLUTION)
                               ↓

PHASE 5: CLUSTER CHARACTERIZATION
═════════════════════════════════════════════════════════════════════
                               ↓
              [9] analyze_subpopulation_proportions()
               ├─ Calculate proportions
               ├─ Chi-square test
               ├─ Post-hoc Z-tests
               └─ FDR correction
                               ↓
            [10] plot_subpopulation_proportions() ◄─ VISUALIZATION
             └─ Stacked bar chart by condition
                               ↓

PHASE 6: STATISTICAL FEATURE ANALYSIS
═════════════════════════════════════════════════════════════════════
                               ↓
                [11] statistical_test_clusters()
                 ├─ Mann-Whitney U tests
                 ├─ Fold change calculation
                 ├─ Multiple testing correction
                 └─ Return significant features
                               ↓
              [12] plot_statistical_results() ◄─ VISUALIZATION
               └─ Dot plot with effect sizes
                               ↓

PHASE 7: FEATURE VALUE VISUALIZATION
═════════════════════════════════════════════════════════════════════
                               ↓
          [13] visualize_feature_values_umap() ◄─ VISUALIZATION
           ├─ For each feature:
           │  ├─ UMAP colored by feature value
           │  ├─ Blue-white-red colormap
           │  └─ Save PNG + SVG
           └─ Generate 20-30+ plots
                               ↓

PHASE 8: FEATURE IMPORTANCE ANALYSIS
═════════════════════════════════════════════════════════════════════
                               ↓
             [14] feature_importance_analysis()
              ├─ Train Random Forest
              ├─ Compute importances
              ├─ Binary (2 clusters) or
              │  One-vs-rest (>2 clusters)
              └─ Return importance matrix
                               ↓
            [15] plot_feature_importance() ◄─ VISUALIZATION
             └─ Importance heatmap
                  (features × clusters)
                               ↓

PHASE 9: SUBPOPULATION FINGERPRINTS
═════════════════════════════════════════════════════════════════════
                               ↓
             [16] subpopulation_fingerprints()
              ├─ Mean feature per cluster
              └─ Generate fingerprint matrix
                               ↓
              [17] plot_fingerprints() ◄─ VISUALIZATION
               ├─ Heatmap with blue-white-green
               ├─ Negative (blue) values
               ├─ Zero (white) band
               └─ Positive (green) values
                               ↓

═════════════════════════════════════════════════════════════════════
                         ANALYSIS COMPLETE ✓
═════════════════════════════════════════════════════════════════════
```

---

## Function Categories

### Core Analysis Functions (8 functions)
Located in: `R/clustering_analysis.R`

```
1. load_and_preprocess_features()
2. filter_correlated_features()
4. perform_umap_analysis()
6. leiden_clustering_optimization()
9. analyze_subpopulation_proportions()
11. statistical_test_clusters()
14. feature_importance_analysis()
16. subpopulation_fingerprints()
```

### Visualization Functions (9 functions)
Located in: `R/clustering_visualization.R`

```
3. plot_correlation_matrix()
5. plot_umap_conditions()
7. plot_leiden_metrics()
8. plot_umap_clusters()
10. plot_subpopulation_proportions()
12. plot_statistical_results()
13. visualize_feature_values_umap()
15. plot_feature_importance()
17. plot_fingerprints()
```

---

## Dependencies Between Functions

```
[1] load_and_preprocess_features
    └─→ returns: data, labels, output_dir
    └─→ input to [2]

[2] filter_correlated_features
    ├─→ returns: filtered_data, corr_original, corr_filtered
    ├─→ input to [3] (visualization)
    ├─→ input to [4]
    └─→ input to [13] (visualization)

[4] perform_umap_analysis
    ├─→ returns: umap_coords, scaler, umap_model
    ├─→ input to [5] (visualization)
    ├─→ input to [6]
    ├─→ input to [8] (visualization)
    └─→ input to [13] (visualization)

[6] leiden_clustering_optimization
    ├─→ returns: resolutions, silhouette_scores, clusters
    ├─→ input to [7] (visualization)
    └─→ input to [8] (visualization)

[9] analyze_subpopulation_proportions
    ├─→ returns: proportions, chi2_pvalue, pairwise_tests
    └─→ input to [10] (visualization)

[11] statistical_test_clusters
    ├─→ returns: results, significant_features
    └─→ input to [12] (visualization)

[14] feature_importance_analysis
    ├─→ returns: importance_matrix, model_stats
    └─→ input to [15] (visualization)

[16] subpopulation_fingerprints
    ├─→ returns: fingerprint_matrix, summary_stats
    └─→ input to [17] (visualization)
```

---

## Input/Output Data Structures

### [1] load_and_preprocess_features()
```
INPUT:  "data.xlsx" (Excel file with features + Class column)
OUTPUT: list(
  data = data.frame (n_samples × n_features + Class),
  labels = character vector (n_samples),
  output_dir = character
)
```

### [2] filter_correlated_features()
```
INPUT:  data.frame (n_samples × n_features + Class)
OUTPUT: list(
  filtered_data = data.frame (n_samples × n_selected_features),
  corr_original = matrix (n_features × n_features),
  corr_filtered = matrix (n_selected × n_selected)
)
```

### [4] perform_umap_analysis()
```
INPUT:  data.frame (n_samples × n_features)
OUTPUT: list(
  umap_coords = data.frame (n_samples × 2, columns: UMAP1, UMAP2),
  scaler = list(center, scale),
  umap_model = umap object
)
```

### [6] leiden_clustering_optimization()
```
INPUT:  UMAP model, coordinates, scaled data matrix
OUTPUT: list(
  resolutions = numeric vector,
  silhouette_scores = numeric vector,
  dbi_scores = numeric vector,
  modularity_scores = numeric vector,
  clusters = integer vector (n_samples),
  partition = igraph partition object
)
```

### [9] analyze_subpopulation_proportions()
```
INPUT:  data.frame with $Clusters and $Condition columns
OUTPUT: list(
  proportions = matrix (n_clusters × n_conditions),
  chi2_pvalue = numeric,
  pairwise_tests = data.frame (cluster_pair × condition_pair results)
)
```

### [11] statistical_test_clusters()
```
INPUT:  data.frame with $Clusters column + feature columns
OUTPUT: list(
  results = data.frame (n_features × n_tests, with p-values, fold-changes),
  significant_features = data.frame (subset where p < 0.05)
)
```

### [14] feature_importance_analysis()
```
INPUT:  data.frame with $Clusters column + feature columns
OUTPUT: list(
  importance_matrix = matrix (n_clusters × n_features),
  model_stats = list(n_clusters, n_features, n_samples)
)
```

### [16] subpopulation_fingerprints()
```
INPUT:  data.frame with $Clusters column + feature columns
OUTPUT: list(
  fingerprint_matrix = matrix (n_clusters × n_features),
  summary_stats = list(n_clusters, n_features, mean_values)
)
```

---

## Recommended Parameter Values

| Function | Parameter | Recommended | Range |
|----------|-----------|-------------|-------|
| [1] load_and_preprocess | qalpha | 0.10 | 0.05-0.20 |
| [1] load_and_preprocess | gamma | 0.05 | 0.01-0.10 |
| [2] filter_correlated | threshold | 0.70 | 0.50-0.90 |
| [4] perform_umap | n_neighbors | 20 | 15-30 |
| [4] perform_umap | min_dist | 1.0 | 0.1-2.0 |
| [4] perform_umap | metric | "correlation" | "euclidean", "cosine" |
| [6] leiden | n_resolutions | 50 | 20-100 |
| [6] leiden | optimal_resolution | 0.004 | 0.0001-0.1 |
| [11] statistical_test | correction | "fdr" | "bonferroni" |
| [14] feature_importance | n_trees | 100 | 50-500 |

---

## Usage in demo.R

The updated `demo.R` file now includes:
✅ All 17 functions in execution order
✅ Proper data flow between functions
✅ All phases of analysis (1-9)
✅ Both analysis and visualization steps
✅ Complete example workflow

To run the demo:
```r
source("inst/demo.R")
```

---

## Translation Summary

| Metric | Count |
|--------|-------|
| **Total Functions** | 17 |
| **Analysis Functions** | 8 |
| **Visualization Functions** | 9 |
| **Python Lines Translated** | 979 |
| **R Lines Generated** | ~1,100 |
| **Documentation Pages** | 5 |
| **Phases of Analysis** | 9 |
| **Statistical Tests** | 3 (Mann-Whitney U, Chi-square, Z-test) |
| **Plotting Libraries** | 3 (ggplot2, pheatmap, base R) |

