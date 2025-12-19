# =============================================================================
# RabAnalyser Demo Script
# =============================================================================
# This script demonstrates the complete RabAnalyser workflow:
# 1. Feature Extraction from microscopy images
# 2. KS Statistical Analysis for comparing conditions
# 3. UMAP Embedding and Resolution Scan
# 4. Leiden Clustering and Stability Analysis
# 5. Statistical Testing and Visualization
# =============================================================================

# Load required libraries
library(RabAnalyser)
library(readr)
library(readxl)
library(dplyr)

# -----------------------------------------------------------------------------
# OPTION 1: Run the Interactive Shiny App
# -----------------------------------------------------------------------------
# Uncomment the line below to launch the interactive web interface
# RabAnalyser::rabanalyser.run()

# -----------------------------------------------------------------------------
# STEP 1: FEATURE EXTRACTION
# -----------------------------------------------------------------------------
# Extract morphological and intensity features from microscopy images
# Adjust paths and parameters according to your data

cat("\n=== STEP 1: FEATURE EXTRACTION ===\n")

# Extract features from image folders
results <- RabAnalyser::extract_features(
  input_folder = "../ProvaIntera/Input_step1/",
  min_spot_size = 8,          # Minimum spot area in pixels
  neighbor_radius = 15,       # Radius for neighbor analysis
  n_jobs = 15,                # Number of parallel jobs
  spot_folder = "rab5_mask",  # Subfolder containing spot masks
  nucleus_folder = "nucleus_mask",
  cell_folder = "cell_mask",
  rab_folder = "Rab5"         # Subfolder with Rab channel images
)

# Save extracted features
# write_csv(results, "../ProvaIntera/extracted_features.csv")

# -----------------------------------------------------------------------------
# STEP 2: KS STATISTICAL ANALYSIS
# -----------------------------------------------------------------------------
# Perform Kolmogorov-Smirnov tests to compare feature distributions
# between reference and test populations

cat("\n=== STEP 2: KS STATISTICAL ANALYSIS ===\n")

# Load reference population (control)
Reference_populationRab5_V2 <- read_excel("~/ProvaIntera/Input_step2/Reference_populationRab5_V2.xlsx")
Reference_populationRab5_V2$ID_image <- 1
Reference_populationRab5_V2 <- Reference_populationRab5_V2 %>% 
  rename(Cell_label = `Cell label`)

# Load test populations
WK1 <- read_csv("~/ProvaIntera/Input_step1/WK1.csv")
FPW1 <- read_csv("~/ProvaIntera/Input_step1/FPW1.csv")
JK2 <- read_csv("~/ProvaIntera/Input_step1/JK2.csv")

# Get feature names (exclude ID columns)
features <- colnames(WK1[, -(1:2)])

# Perform KS analysis for each condition
cat("Analyzing WK1...\n")
df_WK1 <- RabAnalyser::perform_ks_analysis(
  ctrl_matrix = Reference_populationRab5_V2,
  comp_matrix = WK1,
  selected_features = features,
  cores = 10
)
write_csv(df_WK1, file = "../ProvaIntera/PerniceResults/WK1_step2.csv")

cat("Analyzing FPW1...\n")
df_FPW1 <- RabAnalyser::perform_ks_analysis(
  ctrl_matrix = Reference_populationRab5_V2,
  comp_matrix = FPW1,
  selected_features = features,
  cores = 10
)
write_csv(df_FPW1, file = "../ProvaIntera/PerniceResults/FPW1_step2.csv")

cat("Analyzing JK2...\n")
df_JK2 <- RabAnalyser::perform_ks_analysis(
  ctrl_matrix = Reference_populationRab5_V2,
  comp_matrix = JK2,
  selected_features = features,
  cores = 10
)
write_csv(df_JK2, file = "../ProvaIntera/PerniceResults/JK2_step2.csv")

# -----------------------------------------------------------------------------
# STEP 3: DATA CLUSTERING AND SUBPOPULATION ANALYSIS
# -----------------------------------------------------------------------------

cat("\n=== STEP 3: CLUSTERING ANALYSIS ===\n")

# --- PHASE 1: Load Data ---
cat("\nPhase 1: Loading data...\n")
df <- read_excel("~/Desktop/SimoeTealdiDATA/Analisi/Input_step3/GlioCells_KSvaluesRab5WholeRef_V2.xlsx")

# --- PHASE 2: Feature Selection ---
cat("\nPhase 2: Filtering correlated features...\n")
resultCorr <- RabAnalyser::filter_correlated_features(df, threshold = 0.7)

# Visualize correlation matrices
cat("Plotting correlation matrices...\n")
RabAnalyser::plot_correlation_matrix(
  resultCorr$corr_original, 
  title = "Correlation Matrix - Original Features"
)
RabAnalyser::plot_correlation_matrix(
  resultCorr$corr_filtered, 
  title = "Correlation Matrix - Filtered Features"
)

# Save filtered data
write.csv(
  resultCorr$filtered_data, 
  file = "~/Desktop/SimoeTealdiDATA/Analisi/FilteredData.csv",
  row.names = FALSE
)

# --- PHASE 3a: UMAP Embedding and Resolution Scan ---
cat("\nPhase 3a: Running UMAP and resolution scan...\n")
umap_results <- RabAnalyser::run_umap_resolution_scan(
  data = "~/Desktop/SimoeTealdiDATA/Analisi/FilteredData.csv",
  n_neighbors = 15,       # UMAP neighbors parameter
  min_dist = 0.1,         # UMAP minimum distance
  gamma_min = 0.1,        # Minimum resolution to scan
  gamma_max = 1.0,        # Maximum resolution to scan
  n_gamma_steps = 100,    # Number of gamma values to test
  save_graph = TRUE       # Save graph for clustering
)

# Visualize resolution scan
cat("Plotting resolution scan...\n")
RabAnalyser::plot_resolution_scan(umap_results$resolution_scan)

# Visualize UMAP colored by condition
cat("Plotting UMAP by condition...\n")
RabAnalyser::plot_umap(umap_results$umap_df, color_by = "Class", discrete = TRUE)

# --- PHASE 3b: Leiden Clustering with Selected Gamma ---
cat("\nPhase 3b: Running Leiden clustering...\n")
cat("Selected gamma: 0.4\n")

leiden_results <- RabAnalyser::run_leiden_clustering(
  umap_data = umap_results$umap_df,
  graph_path = umap_results$graph_path,
  gamma = 0.4,              # Resolution parameter (chosen from scan)
  n_bootstrap = 100,        # Bootstrap iterations for stability
  subsample_prop = 0.8,     # Subsample 80% of data
  stability_analysis = TRUE
)

# Visualize cluster stability
cat("Plotting cluster stability...\n")
RabAnalyser::plot_cluster_stability(leiden_results$stability)

# Visualize UMAP colored by clusters
cat("Plotting UMAP by clusters...\n")
RabAnalyser::plot_umap(leiden_results$umap_df, color_by = "Cluster", discrete = TRUE)

# -----------------------------------------------------------------------------
# STEP 4: DOWNSTREAM ANALYSIS
# -----------------------------------------------------------------------------

cat("\n=== STEP 4: DOWNSTREAM ANALYSIS ===\n")

# Combine features with cluster assignments
df_features <- resultCorr$filtered_data %>% 
  rename(Condition = Class) %>% 
  mutate(Clusters = leiden_results$umap_df$Cluster)

# --- Subpopulation Proportions ---
cat("\nAnalyzing subpopulation proportions...\n")
proportions_result <- RabAnalyser::analyze_subpopulation_proportions(df_features)
print(proportions_result)

# Plot cluster proportions across conditions
RabAnalyser::plot_clusters_proportions(reduced_df = df_features)

# --- Statistical Testing Between Clusters ---
cat("\nPerforming statistical tests between clusters...\n")
stats_result <- RabAnalyser::cluster_feature_stats(df_features)
stats_result$plot

# --- KS Cluster Fingerprint Heatmap ---
cat("\nGenerating cluster fingerprint heatmap...\n")
fingerprint_result <- RabAnalyser::ks_cluster_fingerprint_heatmap(
  df_features,
  values_interval = c(-0.3, 0.3),
  midpoint_val = 0
)
fingerprint_result$plot

# --- Feature Importance Analysis ---
cat("\nAnalyzing feature importance...\n")
importance_result <- RabAnalyser::feature_importance_analysis(
  df_features %>% select(-Condition)
)
print(importance_result)

# -----------------------------------------------------------------------------
# STEP 5: OPTIONAL - VISUALIZE FEATURES IN UMAP SPACE
# -----------------------------------------------------------------------------

cat("\n=== STEP 5: FEATURE VISUALIZATION (OPTIONAL) ===\n")

# Note: This requires UMAP results with feature values included
# Uncomment if your results include feature columns

# RabAnalyser::plot_umap(
#   results$umap_features, 
#   color_by = "size",
#   high_color = "#228B22",
#   values_interval = c(-0.5, 0.5)
# )
# 
# RabAnalyser::plot_umap(
#   results$umap_features, 
#   color_by = "MeanInt",
#   high_color = "#228B22",
#   values_interval = c(-0.5, 0.5)
# )

cat("\n=== ANALYSIS COMPLETE ===\n")
importanceRes$plot
