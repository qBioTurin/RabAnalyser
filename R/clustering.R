#' Prepare Clustering Data from Uploaded Files
#'
#' Loads Excel files, applies correlation-based feature filtering, and formats data for downstream clustering.
#'
#' This function reads the provided Excel files, cleans the data by removing missing values,
#' and applies a correlation threshold to remove highly correlated features. It returns a list
#' of cleaned and filtered data frames with associated sample names for each file.
#'
#' @param file_paths A character vector of file paths to the uploaded Excel files.
#' @param file_names A character vector of the corresponding file names.
#' @param corr_threshold A numeric threshold (between 0 and 1) to remove highly correlated features. Default is 0.75.
#'
#' @return A named list where each element is a data frame with filtered features ready for clustering.
#' @export
#' @import readxl

prepare_clustering_data <- function(file_paths, file_names, corr_threshold = 0.75) {
  df <- do.call(rbind, lapply(seq_along(file_paths), function(i) {
    temp <- readxl::read_excel(file_paths[i])
    temp$Treatment <- gsub("\\.xlsx$", "", file_names[i])
    temp
  }))

  reduced_df <- filter_features(df %>% dplyr::select(-Treatment), threshold = corr_threshold)
  filtered_data <- cbind(reduced_df, df %>% dplyr::select(Treatment))

  list(
    original_data = df,
    filtered_data = filtered_data,
    features = colnames(reduced_df),
    treatments = unique(df$Treatment)
  )
}

#' Perform UMAP Dimensionality Reduction and Clustering
#'
#' Applies UMAP for dimensionality reduction and k-means clustering on the filtered dataset.
#'
#' This function first standardizes the input data, then applies UMAP to reduce it to 2D.
#' It uses internal clustering indexes to determine the optimal number of clusters (`k`)
#' and performs k-means clustering. The result includes UMAP coordinates and cluster assignments.
#'
#' @param filtered_df A numeric data frame with features already filtered (e.g., via correlation thresholding).
#' @param seed An integer seed for reproducibility of UMAP and k-means. Default is 42.
#'
#' @return A list containing:
#' \describe{
#'   \item{umap_result}{A data frame with UMAP1, UMAP2, and cluster labels.}
#'   \item{k}{The optimal number of clusters found.}
#'   \item{model}{The result of the k-means model.}
#' }
#' @export
#' @import uwot


perform_umap_clustering <- function(filtered_df, seed = 42) {
  scaled_data <- scale(filtered_df, center = TRUE, scale = apply(filtered_df, 2, sd) * sqrt((nrow(filtered_df) - 1) / nrow(filtered_df)))
  set.seed(seed)
  reducer <- uwot::umap(scaled_data, n_neighbors = 10, min_dist = 0.5, n_components = 2)
  umap_df <- as.data.frame(reducer)
  colnames(umap_df) <- c("UMAP1", "UMAP2")

  clustering <- cluster_generation(umap_df)
  return(list(umap_df = umap_df, clustering = clustering))
}
