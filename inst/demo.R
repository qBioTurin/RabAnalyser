
results <- RabAnalyser::extract_features("../ProvaIntera/Input_step1/",
                            min_spot_size = 8,
                            neighbor_radius = 15,
                            n_jobs = 15,
                            spot_folder = "rab5_mask",
                            nucleus_folder = "nucleus_mask",
                            cell_folder = "cell_mask",
                            rab_folder = "Rab5"
                          )

##### START: SC_KS_singlePopulation #####

library(readr)
library(readxl)
Reference_populationRab5_V2 <- read_excel("~/ProvaIntera/Input_step2/Reference_populationRab5_V2.xlsx")
Reference_populationRab5_V2$ID_image = 1
Reference_populationRab5_V2 = Reference_populationRab5_V2 %>% rename(Cell_label =`Cell label`)
WK1 <- read_csv("~/ProvaIntera/Input_step1/WK1.csv")

features = colnames(WK1[,-(1:2)])

df_WK1 = RabAnalyser::perform_ks_analysis(ctrl_matrix = Reference_populationRab5_V2,
                                      comp_matrix = WK1,
                                      features,
                                      cores = 10)
write_csv(df_WK1,file = "../ProvaIntera/PerniceResults/WK1_step2.csv")

FPW1 <- read_csv("~/ProvaIntera/Input_step1/FPW1.csv")
df_FPW1 = RabAnalyser::perform_ks_analysis(ctrl_matrix = Reference_populationRab5_V2,
                                          comp_matrix = FPW1,
                                          features,
                                          cores = 10)
write_csv(df_FPW1,file = "../ProvaIntera/PerniceResults/FPW1_step2.csv")

JK2 <- read_csv("~/ProvaIntera/Input_step1/JK2.csv")
df_JK2 = RabAnalyser::perform_ks_analysis(ctrl_matrix = Reference_populationRab5_V2,
                                          comp_matrix = JK2,
                                          features,
                                          cores = 10)
write_csv(df_JK2,file = "../ProvaIntera/PerniceResults/JK2_step2.csv")



##### START: SC_MultipleCNDTN_Analysis (Single-Cell Subpopulation Analysis) #####

# PHASE 1: Load & Preprocess
# Function 1: Load data and apply soft-threshold denoising
df <- read_excel("~/ProvaIntera/Input_step3/GlioCells_KSvaluesRab5WholeRef_V2.xlsx")

# PHASE 2: Feature Selection
# Function 2: Filter correlated features
resultCorr <- RabAnalyser::filter_correlated_features(df, threshold = 0.7)

# Function 3 (Visualization): Plot correlation matrices
RabAnalyser::plot_correlation_matrix(resultCorr$corr_original, title = "Correlation Matrix - Original Features")
RabAnalyser::plot_correlation_matrix(resultCorr$corr_filtered, title = "Correlation Matrix - Filtered Features")

# PHASE 3: Leiden Clustering Optimization
write.csv(file = "../ProvaIntera/PerniceResults/FilteredData.csv", x = resultCorr$filtered_data,row.names = F)

results <- RabAnalyser::run_umap_leiden(
  data_path = "../ProvaIntera/PerniceResults/FilteredData.csv",
  n_neighbors = 15,
  min_dist = 0.1,
  resolution = 0.42,
  n_bootstrap = 100
)

RabAnalyser::plot_resolution_scan(results$resolution_scan)
RabAnalyser::plot_cluster_stability(results$stability)
RabAnalyser::plot_umap(results$umap_df, color_by = "Cluster",discrete = T)
RabAnalyser::plot_umap(results$umap_df, color_by = "Class")


df_features = resultCorr$filtered_data %>% rename(Condition = Class) %>% mutate(Clusters = results$umap_df$Cluster)

result <- RabAnalyser::analyze_subpopulation_proportions(df_features)
RabAnalyser::plot_clusters_proportions(df_features)

### FEATURES VALUES VISUALIZATION IN UMAP ###
umap_res = cbind(results$umap_df, resultCorr$filtered_data %>% select(-Class))
RabAnalyser::plot_umap(umap_res, color_by = "size",high_color = "#228B22",values_interval = c(-.5,.5))
RabAnalyser::plot_umap(umap_res, color_by = "MeanInt",high_color = "#228B22",values_interval = c(-.5,.5))

### STATISTICAL TEST  ###
statsRes = RabAnalyser::cluster_feature_stats(df_features)
statsRes$plot

fingerprintRes = RabAnalyser::ks_cluster_fingerprint_heatmap(df_features,values_interval = c(-0.3, 0.3),midpoint_val = 0 )
fingerprintRes$plot

importanceRes = RabAnalyser::feature_importance_analysis(df_features %>% select(-Condition))
importanceRes$plot
