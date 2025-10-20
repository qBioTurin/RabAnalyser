# Complete Ordered List of Translated Functions
## From SC_MultipleCNDTN_Analysis_V3.py → RabAnalyser R Package

### Execution Order for Complete Single-Cell Analysis Workflow

Based on `demo.R` and the complete Python script, here is the logical execution sequence:

---

## **PHASE 1: Data Loading & Preprocessing**

### 1. `load_and_preprocess_features()`
**Location**: `R/clustering_analysis.R`  
**Replaces Python**: Lines 1-90 (File loading, soft-thresholding, subsampling)  
**Purpose**: Load Excel/CSV, apply soft-threshold denoising, subsample to equal group sizes  
**Input**: Path to feature matrix Excel/CSV file  
**Output**: Preprocessed data frame, labels, output directory  

```r
result <- load_and_preprocess_features(
  "~/Desktop/SimoeTealdiDATA/Images/Vehicle_vs_CET_Rab5_KS.csv",
  qalpha = 0.10, 
  gamma = 0.05
)
df <- result$data
labels <- result$labels
```

---

## **PHASE 2: Feature Selection**

### 2. `filter_correlated_features()`
**Location**: `R/clustering_analysis.R`  
**Replaces Python**: Lines 100-170 (Feature selection, correlation matrices)  
**Purpose**: Remove redundant correlated features, compute correlation matrices  
**Input**: Data frame with numeric features  
**Output**: Filtered data, original & filtered correlation matrices  

```r
resultCorr <- filter_correlated_features(
  result$data, 
  threshold = 0.7
)
df_filtered <- resultCorr$filtered_data
corr_original <- resultCorr$corr_original
corr_filtered <- resultCorr$corr_filtered
```

### 3. `plot_correlation_matrix()` (VISUALIZATION)
**Location**: `R/clustering_visualization.R`  
**Replaces Python**: Lines 112-170 (Correlation heatmap plots)  
**Purpose**: Create heatmaps of before/after feature correlations  

```r
plot_correlation_matrix(
  resultCorr$corr_original, 
  title = "Original Features"
)
plot_correlation_matrix(
  resultCorr$corr_filtered, 
  title = "Filtered Features"
)
```

---

## **PHASE 3: Dimensionality Reduction**

### 4. `perform_umap_analysis()`
**Location**: `R/clustering_analysis.R`  
**Replaces Python**: Lines 175-225 (UMAP standardization and projection)  
**Purpose**: Standardize features, compute 2D UMAP projection  
**Input**: Filtered feature data frame  
**Output**: UMAP coordinates, scaler info, UMAP model  

```r
umap_result <- perform_umap_analysis(
  resultCorr$filtered_data, 
  n_neighbors = 20, 
  min_dist = 1,
  metric = "correlation"
)
umap_coords <- umap_result$umap_coords
```

### 5. `plot_umap_conditions()` (VISUALIZATION)
**Location**: `R/clustering_visualization.R`  
**Replaces Python**: Lines 226-275 (UMAP colored by condition)  
**Purpose**: Visualize UMAP with condition colors  

```r
plot_umap_conditions(
  umap_result$umap_coords, 
  result$labels
)
```

---

## **PHASE 4: Clustering Optimization**

### 6. `leiden_clustering_optimization()`
**Location**: `R/clustering_analysis.R`  
**Replaces Python**: Lines 280-380 (Leiden resolution sweep, metric computation)  
**Purpose**: Test multiple Leiden resolutions, compute clustering metrics  
**Input**: UMAP model, coordinates, scaled data  
**Output**: Resolution values, silhouette/DBI/modularity scores, optimal clusters  

```r
leiden_result <- leiden_clustering_optimization(
  umap_result$umap_model,
  umap_result$umap_coords,
  scale(resultCorr$filtered_data),
  resolution_range = c(0.0001, 0.1),
  n_resolutions = 50,
  optimal_resolution = 0.004
)
```

### 7. `plot_leiden_metrics()` (VISUALIZATION)
**Location**: `R/clustering_visualization.R`  
**Replaces Python**: Lines 300-380 (Silhouette, DBI, modularity plots)  
**Purpose**: Visualize clustering quality metrics vs. resolution  

```r
plot_leiden_metrics(
  leiden_result, 
  output_dir = "."
)
```

### 8. `plot_umap_clusters()` (VISUALIZATION)
**Location**: `R/clustering_visualization.R`  
**Replaces Python**: Lines 400-450 (UMAP colored by cluster)  
**Purpose**: Visualize UMAP with cluster assignments  

```r
plot_umap_clusters(
  umap_result$umap_coords, 
  leiden_result$clusters,
  save_path = "UMAP_clusters.png"
)
```

---

## **PHASE 5: Cluster Characterization**

### 9. `analyze_subpopulation_proportions()`
**Location**: `R/clustering_analysis.R`  
**Replaces Python**: Lines 460-550 (Proportions & chi-square test)  
**Purpose**: Calculate cell proportions per cluster/condition, test for differences  
**Input**: Data frame with clusters and conditions  
**Output**: Proportions matrix, chi-square p-value, pairwise test results  

```r
# First add cluster and condition columns
df_with_clusters <- resultCorr$filtered_data
df_with_clusters$Clusters <- leiden_result$clusters
df_with_clusters$Condition <- result$labels

prop_result <- analyze_subpopulation_proportions(
  df_with_clusters
)
```

### 10. `plot_subpopulation_proportions()` (VISUALIZATION)
**Location**: `R/clustering_visualization.R`  
**Replaces Python**: Lines 460-550 (Stacked bar plot)  
**Purpose**: Visualize cell proportions across conditions  

```r
plot_subpopulation_proportions(
  prop_result$proportions,
  save_path = "proportions.png"
)
```

---

## **PHASE 6: Statistical Feature Analysis**

### 11. `statistical_test_clusters()`
**Location**: `R/clustering_analysis.R`  
**Replaces Python**: Lines 560-700 (Mann-Whitney U tests with correction)  
**Purpose**: Test for feature differences between clusters  
**Input**: Data frame with features and clusters  
**Output**: Full results table, significant features subset  

```r
stat_result <- statistical_test_clusters(
  df_with_clusters,
  clusters_col = "Clusters",
  correction = "fdr"
)
```

### 12. `plot_statistical_results()` (VISUALIZATION)
**Location**: `R/clustering_visualization.R`  
**Replaces Python**: Lines 700-750 (Dot plot with effect sizes)  
**Purpose**: Visualize test results as dot plot  

```r
plot_statistical_results(
  stat_result$results,
  save_path = "statistics.png"
)
```

---

## **PHASE 7: Feature Value Visualization**

### 13. `visualize_feature_values_umap()`
**Location**: `R/clustering_visualization.R`  
**Replaces Python**: Lines 760-850 (Feature value heatmaps on UMAP)  
**Purpose**: Create UMAP plot colored by each feature value  
**Input**: UMAP coordinates, feature data frame  
**Output**: List of ggplot objects, PNG/SVG files per feature  

```r
visualize_feature_values_umap(
  umap_result$umap_coords,
  resultCorr$filtered_data,
  output_dir = ".",
  save_plots = TRUE,
  vmin = -0.4,
  vmax = 0.4
)
```

---

## **PHASE 8: Feature Importance Analysis**

### 14. `feature_importance_analysis()`
**Location**: `R/clustering_analysis.R`  
**Replaces Python**: Lines 860-950 (Random Forest classification)  
**Purpose**: Compute Random Forest feature importances per cluster  
**Input**: Data frame with features and clusters  
**Output**: Feature importance matrix, model statistics  

```r
importance_result <- feature_importance_analysis(
  df_with_clusters,
  clusters_col = "Clusters",
  n_trees = 100
)
```

### 15. `plot_feature_importance()` (VISUALIZATION)
**Location**: `R/clustering_visualization.R`  
**Replaces Python**: Lines 900-950 (Importance heatmap)  
**Purpose**: Create heatmap of feature importances  

```r
plot_feature_importance(
  importance_result$importance_matrix,
  save_path = "feature_importance.png"
)
```

---

## **PHASE 9: Subpopulation Fingerprints**

### 16. `subpopulation_fingerprints()`
**Location**: `R/clustering_analysis.R`  
**Replaces Python**: Lines 960-979 (Mean feature values per cluster)  
**Purpose**: Compute mean feature profile per subpopulation  
**Input**: Data frame with features and clusters  
**Output**: Fingerprint matrix (clusters × features), summary stats  

```r
fingerprint_result <- subpopulation_fingerprints(
  df_with_clusters,
  clusters_col = "Clusters"
)
```

### 17. `plot_fingerprints()` (VISUALIZATION)
**Location**: `R/clustering_visualization.R`  
**Replaces Python**: Lines 960-979 (Custom colored heatmap)  
**Purpose**: Create heatmap with blue-white-green colors for fingerprints  

```r
plot_fingerprints(
  fingerprint_result$fingerprint_matrix,
  save_path = "fingerprints.png"
)
```

---

## **COMPLETE WORKFLOW EXAMPLE**

```r
library(RabAnalyser)

# PHASE 1: Load & preprocess
result <- load_and_preprocess_features(
  "~/data/features.xlsx",
  qalpha = 0.10, gamma = 0.05
)

# PHASE 2: Feature selection
resultCorr <- filter_correlated_features(result$data, threshold = 0.7)
plot_correlation_matrix(resultCorr$corr_original)
plot_correlation_matrix(resultCorr$corr_filtered)

# PHASE 3: UMAP
umap_result <- perform_umap_analysis(resultCorr$filtered_data)
plot_umap_conditions(umap_result$umap_coords, result$labels)

# PHASE 4: Clustering
leiden_result <- leiden_clustering_optimization(
  umap_result$umap_model,
  umap_result$umap_coords,
  scale(resultCorr$filtered_data),
  optimal_resolution = 0.004
)
plot_leiden_metrics(leiden_result)
plot_umap_clusters(umap_result$umap_coords, leiden_result$clusters)

# PHASE 5: Add clusters to data
df_with_clusters <- resultCorr$filtered_data
df_with_clusters$Clusters <- leiden_result$clusters
df_with_clusters$Condition <- result$labels

# PHASE 5: Proportions
prop_result <- analyze_subpopulation_proportions(df_with_clusters)
plot_subpopulation_proportions(prop_result$proportions)

# PHASE 6: Statistics
stat_result <- statistical_test_clusters(df_with_clusters)
plot_statistical_results(stat_result$results)

# PHASE 7: Feature visualization
visualize_feature_values_umap(
  umap_result$umap_coords,
  resultCorr$filtered_data,
  output_dir = "."
)

# PHASE 8: Feature importance
importance_result <- feature_importance_analysis(df_with_clusters)
plot_feature_importance(importance_result$importance_matrix)

# PHASE 9: Fingerprints
fingerprint_result <- subpopulation_fingerprints(df_with_clusters)
plot_fingerprints(fingerprint_result$fingerprint_matrix)
```

---

## **QUICK REFERENCE: FUNCTION MAPPING**

| Order | Function | Type | Python Lines | File |
|-------|----------|------|--------------|------|
| 1 | `load_and_preprocess_features()` | Analysis | 1-90 | clustering_analysis.R |
| 2 | `filter_correlated_features()` | Analysis | 100-170 | clustering_analysis.R |
| 3 | `plot_correlation_matrix()` | Viz | 112-170 | clustering_visualization.R |
| 4 | `perform_umap_analysis()` | Analysis | 175-225 | clustering_analysis.R |
| 5 | `plot_umap_conditions()` | Viz | 226-275 | clustering_visualization.R |
| 6 | `leiden_clustering_optimization()` | Analysis | 280-380 | clustering_analysis.R |
| 7 | `plot_leiden_metrics()` | Viz | 300-380 | clustering_visualization.R |
| 8 | `plot_umap_clusters()` | Viz | 400-450 | clustering_visualization.R |
| 9 | `analyze_subpopulation_proportions()` | Analysis | 460-550 | clustering_analysis.R |
| 10 | `plot_subpopulation_proportions()` | Viz | 460-550 | clustering_visualization.R |
| 11 | `statistical_test_clusters()` | Analysis | 560-700 | clustering_analysis.R |
| 12 | `plot_statistical_results()` | Viz | 700-750 | clustering_visualization.R |
| 13 | `visualize_feature_values_umap()` | Viz | 760-850 | clustering_visualization.R |
| 14 | `feature_importance_analysis()` | Analysis | 860-950 | clustering_analysis.R |
| 15 | `plot_feature_importance()` | Viz | 900-950 | clustering_visualization.R |
| 16 | `subpopulation_fingerprints()` | Analysis | 960-979 | clustering_analysis.R |
| 17 | `plot_fingerprints()` | Viz | 960-979 | clustering_visualization.R |

---

## **NOTES**

- **Analysis Functions** (8 total): Core computational functions in `clustering_analysis.R`
- **Visualization Functions** (9 total): All plotting functions in `clustering_visualization.R`
- **Execution Order**: Numbers 1-17 represent the logical workflow progression
- **Interdependencies**: Each function depends on outputs from previous ones
- **Optional**: All visualization functions are optional for analysis but recommended for interpretation
- **Customizable**: All functions accept parameters for fine-tuning (thresholds, colors, etc.)

---

## **INTEGRATION WITH demo.R**

The demo.R currently shows:
1. ✅ Feature extraction (`extract_features()`)
2. ✅ KS analysis (`perform_ks_analysis()`)
3. ✅ Loading preprocessed data (`load_and_preprocess_features()`)
4. ✅ Correlation filtering (`filter_correlated_features()`)
5. ✅ UMAP analysis (`perform_umap_analysis()`)

**Next steps to complete demo.R**:
- Add Leiden clustering optimization
- Add cluster visualization
- Add statistical testing
- Add feature importance analysis
- Add fingerprint visualization

Would you like me to update `demo.R` to include all these steps?
