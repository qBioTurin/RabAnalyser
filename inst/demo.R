
debug(extract_features)
results <- extract_features("~/Desktop/SimoeTealdiDATA/Images/",
                            min_spot_size = 8,
                            neighbor_radius = 15,
                            n_jobs = 2)

##### START: SC_KS_singlePopulation #####

library(readr)
CET_Rab5 <- read_csv("~/Desktop/SimoeTealdiDATA/Images/CET_Rab5.csv")
Vehicle_Rab5 <- read_csv("~/Desktop/SimoeTealdiDATA/Images/Vehicle_Rab5.csv")

features <- colnames(Vehicle_Rab5)[-1]
# 150 seconds


df = RabAnalyser::perform_ks_analysis(ctrl_matrix = CET_Rab5,
                                      comp_matrix = Vehicle_Rab5,
                                      features,
                                      cores = 2)

profvis::profvis(RabAnalyser::perform_ks_analysis(ctrl_matrix = ctrl[ctrl[,1] %in% 1:150, ],
                                                  comp_matrix = comp[comp[,1] %in% 1:50,],
                                                  features,cores = 1))

##### END: SC_KS_singlePopulation #####

write.csv(df, "~/Desktop/SimoeTealdiDATA/Images/Vehicle_vs_CET_Rab5_KS.csv", row.names = FALSE)

##### START: SC_MultipleCNDTN_Analysis (Single-Cell Subpopulation Analysis) #####

# PHASE 1: Load & Preprocess
# Function 1: Load data and apply soft-threshold denoising
result <- load_and_preprocess_features(
  "~/Desktop/SimoeTealdiDATA/Images/Vehicle_vs_CET_Rab5_KS.csv",
  qalpha = 0.10, 
  gamma = 0.05
)
df <- result$data
labels <- result$labels

# PHASE 2: Feature Selection
# Function 2: Filter correlated features
resultCorr <- filter_correlated_features(result$data, threshold = 0.7)
df_filtered <- resultCorr$filtered_data

# Function 3 (Visualization): Plot correlation matrices
plot_correlation_matrix(resultCorr$corr_original, title = "Correlation Matrix - Original Features")
plot_correlation_matrix(resultCorr$corr_filtered, title = "Correlation Matrix - Filtered Features")

# PHASE 3: Dimensionality Reduction
# Function 4: Perform UMAP analysis
umap_result <- perform_umap_analysis(
  resultCorr$filtered_data, 
  n_neighbors = 20, 
  min_dist = 1,
  metric = "correlation"
)
umap_coords <- umap_result$umap_coords

# Function 5 (Visualization): Plot UMAP by condition
plot_umap_conditions(umap_result$umap_coords, result$labels)

# PHASE 4: Leiden Clustering Optimization
# Function 6: Optimize Leiden resolution
leiden_result <- leiden_clustering_optimization(
  umap_result$umap_model,
  umap_result$umap_coords,
  scale(resultCorr$filtered_data),
  resolution_range = c(0.0001, 0.1),
  n_resolutions = 50,
  optimal_resolution = 0.004
)

# Function 7 (Visualization): Plot clustering metrics
plot_leiden_metrics(leiden_result, output_dir = ".")

# Function 8 (Visualization): Plot UMAP with clusters
plot_umap_clusters(
  umap_result$umap_coords, 
  leiden_result$clusters,
  save_path = "UMAP_clusters.png"
)

# PHASE 5: Cluster Characterization
# Prepare data with clusters and conditions
df_with_clusters <- resultCorr$filtered_data
df_with_clusters$Clusters <- leiden_result$clusters
df_with_clusters$Condition <- result$labels

# Function 9: Analyze subpopulation proportions
prop_result <- analyze_subpopulation_proportions(df_with_clusters)

# Function 10 (Visualization): Plot proportions
plot_subpopulation_proportions(
  prop_result$proportions,
  save_path = "proportions.png"
)

# PHASE 6: Statistical Feature Analysis
# Function 11: Perform Mann-Whitney U tests
stat_result <- statistical_test_clusters(
  df_with_clusters,
  clusters_col = "Clusters",
  correction = "fdr"
)

# Function 12 (Visualization): Plot statistical results
plot_statistical_results(
  stat_result$results,
  save_path = "statistics.png"
)

# PHASE 7: Feature Value Visualization
# Function 13 (Visualization): Visualize features on UMAP
visualize_feature_values_umap(
  umap_result$umap_coords,
  resultCorr$filtered_data,
  output_dir = ".",
  save_plots = TRUE,
  vmin = -0.4,
  vmax = 0.4
)

# PHASE 8: Feature Importance Analysis
# Function 14: Compute Random Forest importance
importance_result <- feature_importance_analysis(
  df_with_clusters,
  clusters_col = "Clusters",
  n_trees = 100
)

# Function 15 (Visualization): Plot importance heatmap
plot_feature_importance(
  importance_result$importance_matrix,
  save_path = "feature_importance.png"
)

# PHASE 9: Subpopulation Fingerprints
# Function 16: Compute fingerprints
fingerprint_result <- subpopulation_fingerprints(
  df_with_clusters,
  clusters_col = "Clusters"
)

# Function 17 (Visualization): Plot fingerprints
plot_fingerprints(
  fingerprint_result$fingerprint_matrix,
  save_path = "fingerprints.png"
)

##### END: SC_MultipleCNDTN_Analysis #####
