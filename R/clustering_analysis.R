#' Load and Preprocess Feature Data
#'
#' Loads feature data from Excel/CSV, applies soft-threshold denoising,
#' and subsamples to equal group sizes for unbiased clustering analysis.
#'
#' @param data_path Character. Path to Excel or CSV file with features.
#'                  Last column should contain condition labels.
#' @param qalpha Numeric. Soft-threshold noise floor (default 0.10).
#' @param gamma Numeric. Soft-threshold steepness parameter (default 0.05).
#'
#' @return List containing:
#'   \describe{
#'     \item{data}{Preprocessed data frame with soft-thresholded features}
#'     \item{labels}{Condition labels vector}
#'     \item{output_dir}{Directory where plots will be saved}
#'   }
#'
#' @details
#' The soft-threshold formula applied is:
#' sign(x) * (|x| - qalpha) * tanh((|x| - qalpha) / gamma)
#'
#' This reduces noise while preserving large signals.
#'
#' @examples
#' \dontrun{
#'   result <- load_and_preprocess_features("data/features.xlsx")
#'   df <- result$data
#'   labels <- result$labels
#' }
#'
#' @import dplyr
#' @export
load_and_preprocess_features <- function(data_path, qalpha = 0.10, gamma = 0.05) {
  # Load data
  if (grepl("\\.xlsx$", data_path)) {
    df <- readxl::read_excel(data_path)
  } else if (grepl("\\.csv$", data_path)) {
    df <- read.csv(data_path)
  } else {
    stop("File must be .xlsx or .csv")
  }

  # Extract labels (last column)
  labels <- df[[ncol(df)]]
  df_num <- df[-ncol(df)]

  # Apply soft-threshold denoising
  df_num <- apply(df_num, 2, function(col) {
    sign(col) * (abs(col) - qalpha) * tanh((abs(col) - qalpha) / gamma)
  })

  df_num <- as.data.frame(df_num)

  # Combine
  df <- cbind(df_num, Class = labels)

  # Subsample to equal counts
  min_count <- min(table(df$Class))
  df <- df %>%
    dplyr::group_by(.data$Class) %>%
    dplyr::slice_sample(n = min_count, replace = FALSE) %>%
    dplyr::ungroup() %>%
    as.data.frame()

  output_dir <- dirname(data_path)

  list(
    data = df,
    labels = labels,
    output_dir = output_dir
  )
}


#' Filter Correlated Features
#'
#' Removes features with high correlation to reduce redundancy.
#' Returns both original and filtered correlation matrices.
#'
#' @param df Data frame with numeric features (last column should be labels).
#' @param threshold Numeric. Pearson correlation threshold for filtering (default 0.7).
#'
#' @return List containing:
#'   \describe{
#'     \item{filtered_data}{Reduced feature set (uncorrelated features)}
#'     \item{corr_original}{Correlation matrix of original features}
#'     \item{corr_filtered}{Correlation matrix of filtered features}
#'   }
#'
#' @details
#' Features are removed if they have correlation > threshold with previously retained features.
#' This greedy approach keeps the first feature and removes subsequent correlated ones.
#'
#' @examples
#' \dontrun{
#'   result <- filter_correlated_features(df_num)
#'   df_filtered <- result$filtered_data
#' }
#'
#' @export
filter_correlated_features <- function(df, threshold = 0.7) {
  # Extract numeric columns (remove labels)
  df_num <- df[, -which(names(df) == "Class")]

  # Original correlation matrix
  corr_original <- cor(df_num, use = "complete.obs")

  # Greedy correlation-based feature selection
  selected_features <- c()
  for (feat in colnames(df_num)) {
    if (length(selected_features) == 0) {
      selected_features <- c(selected_features, feat)
    } else {
      correlations <- abs(cor(df_num[[feat]], df_num[selected_features], use = "complete.obs"))
      if (all(correlations < threshold)) {
        selected_features <- c(selected_features, feat)
      }
    }
  }

  # Filtered data
  df_filtered <- df_num[, selected_features]

  # Filtered correlation matrix
  corr_filtered <- cor(df_filtered, use = "complete.obs")

  list(
    filtered_data = df_filtered,
    corr_original = corr_original,
    corr_filtered = corr_filtered
  )
}


#' Perform UMAP Analysis
#'
#' Standardizes features and computes UMAP projection for 2D visualization.
#'
#' @param df Data frame with numeric features.
#' @param n_neighbors Integer. Number of neighbors for UMAP (default 20).
#' @param min_dist Numeric. Minimum distance for UMAP (default 1).
#' @param metric Character. Distance metric for UMAP (default "correlation").
#' @param seed Integer. Random seed for reproducibility (default 42).
#'
#' @return List containing:
#'   \describe{
#'     \item{umap_coords}{Data frame with UMAP1 and UMAP2 coordinates}
#'     \item{scaler}{Scaling model for standardization}
#'     \item{umap_model}{Fitted UMAP model}
#'   }
#'
#' @examples
#' \dontrun{
#'   result <- perform_umap_analysis(df_filtered)
#'   umap_df <- result$umap_coords
#' }
#'
#' @export
#' @import uwot
perform_umap_analysis <- function(df, n_neighbors = 20, min_dist = 1,
                                   metric = "correlation", seed = 42) {
  # Standardize
  df_scaled <- scale(df)

  # UMAP
  set.seed(seed)
  umap_result <- uwot::umap(df_scaled,
    n_neighbors = n_neighbors,
    min_dist = min_dist,
    metric = metric
  )

  umap_coords <- as.data.frame(umap_result$layout)
  colnames(umap_coords) <- c("UMAP1", "UMAP2")

  list(
    umap_coords = umap_coords,
    scaler = list(center = attr(df_scaled, "scaled:center"),
                  scale = attr(df_scaled, "scaled:scale")),
    umap_model = umap_result
  )
}


#' Leiden Clustering with Resolution Optimization
#'
#' Tests multiple resolution parameters for Leiden clustering and
#' computes clustering quality metrics (silhouette, DBI, modularity).
#'
#' @param umap_model UMAP model object (from perform_umap_analysis).
#' @param umap_coords Data frame with UMAP coordinates.
#' @param data_scaled Scaled feature matrix.
#' @param resolution_range Numeric vector. Range of resolutions to test (default 0.0001-0.1).
#' @param n_resolutions Integer. Number of resolution values to test (default 50).
#' @param optimal_resolution Numeric. Which resolution to use for final clustering
#'                           (if NULL, user should choose based on metrics).
#' @param seed Integer. Random seed for reproducibility (default 42).
#'
#' @return List containing:
#'   \describe{
#'     \item{resolutions}{Resolution values tested}
#'     \item{silhouette_scores}{Silhouette scores for each resolution}
#'     \item{dbi_scores}{Davies-Bouldin Index scores}
#'     \item{modularity_scores}{Modularity scores}
#'     \item{num_clusters}{Number of clusters for each resolution}
#'     \item{clusters}{Final cluster assignments (if optimal_resolution specified)}
#'     \item{partition}{igraph partition object for final clustering}
#'   }
#'
#' @details
#' Silhouette Score: Higher is better (-1 to 1, max is 1)
#' DBI Score: Lower is better (0 to infinity, min is 0)
#' Modularity: Higher is better (0 to 1, max is 1)
#'
#' @examples
#' \dontrun{
#'   result <- leiden_clustering_optimization(
#'     umap_model, umap_coords, data_scaled,
#'     optimal_resolution = 0.004
#'   )
#'   clusters <- result$clusters
#' }
#'
#' @export
#' @import FNN igraph cluster leiden
leiden_clustering_optimization <- function(umap_model, umap_coords, data_scaled,
                                            resolution_range = c(0.0001, 0.1),
                                            n_resolutions = 50,
                                            optimal_resolution = NULL,
                                            seed = 42) {
  # Create graph from UMAP
  # Extract distances and build graph
  # For simplicity, use k-nearest neighbors graph
  k <- 20
  knn_graph <- FNN::get.knn(umap_coords, k = k)$nn.index

  # Convert to edge list
  edges <- list()
  for (i in seq_len(nrow(knn_graph))) {
    for (j in knn_graph[i, ]) {
      edges[[length(edges) + 1]] <- c(i - 1, j - 1) # 0-indexed for igraph
    }
  }

  g <- igraph::graph_from_edgelist(do.call(rbind, edges), directed = FALSE)
  g <- igraph::simplify(g)

  # Test resolutions
  resolutions <- seq(resolution_range[1], resolution_range[2], length.out = n_resolutions)

  silhouette_scores <- c()
  dbi_scores <- c()
  modularity_scores <- c()
  num_clusters <- c()

  set.seed(seed)

  for (res in resolutions) {
    partition <- leiden::leiden(g, resolution_parameter = res)

    labels <- as.numeric(partition) - 1 # Convert to 0-indexed

    n_clust <- length(unique(labels))
    num_clusters <- c(num_clusters, n_clust)

    # Modularity
    modularity_scores <- c(modularity_scores, igraph::modularity(g, labels + 1))

    # Silhouette and DBI
    if (n_clust > 1) {
      sil_score <- cluster::silhouette(labels, dist(data_scaled))[, "sil_width"]
      silhouette_scores <- c(silhouette_scores, mean(sil_score))

      dbi_score <- clusterCrit::intCriteria(data_scaled, labels + 1, "davies_bouldin")[[1]]
      dbi_scores <- c(dbi_scores, dbi_score)
    } else {
      silhouette_scores <- c(silhouette_scores, -1)
      dbi_scores <- c(dbi_scores, Inf)
    }
  }

  result <- list(
    resolutions = resolutions,
    silhouette_scores = silhouette_scores,
    dbi_scores = dbi_scores,
    modularity_scores = modularity_scores,
    num_clusters = num_clusters
  )

  # If optimal resolution specified, compute final clusters
  if (!is.null(optimal_resolution)) {
    set.seed(seed)
    final_partition <- leiden::leiden(g, resolution_parameter = optimal_resolution)
    result$clusters <- as.numeric(final_partition) - 1
    result$partition <- final_partition
  }

  result
}


#' Analyze Subpopulation Proportions
#'
#' Calculates proportions of cells in each cluster by condition and
#' performs statistical tests for differences between conditions.
#'
#' @param df_features Data frame with features, clusters, and condition labels.
#'                    Must have columns "Clusters" and "Condition".
#' @param clusters_col Character. Name of cluster column (default "Clusters").
#' @param condition_col Character. Name of condition column (default "Condition").
#'
#' @return List containing:
#'   \describe{
#'     \item{proportions}{Data frame with proportions by cluster and condition}
#'     \item{chi2_pvalue}{Global chi-square test p-value}
#'     \item{pairwise_tests}{Data frame with pairwise post-hoc test results}
#'   }
#'
#' @details
#' Performs global chi-square test. If significant (p < 0.05), proceeds with
#' pairwise Z-tests or Fisher's exact tests (depending on expected counts).
#' P-values adjusted using FDR method.
#'
#' @examples
#' \dontrun{
#'   df_features$Clusters <- clusters
#'   df_features$Condition <- conditions
#'   result <- analyze_subpopulation_proportions(df_features)
#' }
#'
#' @export
analyze_subpopulation_proportions <- function(df_features,
                                               clusters_col = "Clusters",
                                               condition_col = "Condition") {
  # Contingency table
  contingency <- table(df_features[[clusters_col]], df_features[[condition_col]])

  # Global chi-square test
  chi2_result <- chisq.test(contingency)

  cat(sprintf("\nGlobal Chi-square test p-value = %.4e\n\n", chi2_result$p.value))

  pairwise_results <- NULL

  if (chi2_result$p.value < 0.05) {
    cat("Global test significant. Proceeding with pairwise post-hoc tests...\n\n")

    clusters <- rownames(contingency)
    conditions <- colnames(contingency)
    condition_pairs <- combn(conditions, 2, simplify = FALSE)

    raw_pvalues <- c()
    test_records <- list()

    for (cluster in clusters) {
      for (pair in condition_pairs) {
        cond1 <- pair[1]
        cond2 <- pair[2]

        count <- c(contingency[cluster, cond1], contingency[cluster, cond2])
        nobs <- c(sum(contingency[, cond1]), sum(contingency[, cond2]))

        # Z-test if expected > 5, Fisher otherwise
        if (all(count >= 5)) {
          test <- prop.test(count, nobs)
          p <- test$p.value
        } else {
          # Fisher's exact test
          contingency_2x2 <- matrix(c(
            count[1], nobs[1] - count[1],
            count[2], nobs[2] - count[2]
          ), nrow = 2, byrow = TRUE)
          test <- fisher.test(contingency_2x2)
          p <- test$p.value
        }

        raw_pvalues <- c(raw_pvalues, p)
        test_records[[length(test_records) + 1]] <- list(
          Cluster = cluster,
          Condition_1 = cond1,
          Condition_2 = cond2,
          RawPvalue = p
        )
      }
    }

    # Adjust p-values (FDR)
    adjusted_p <- p.adjust(raw_pvalues, method = "fdr")

    pairwise_results <- do.call(rbind, lapply(test_records, as.data.frame))
    pairwise_results$AdjustedPvalue <- adjusted_p
    pairwise_results$Significant <- ifelse(adjusted_p < 0.05, "Yes", "No")

    cat("Pairwise Post Hoc Test Results:\n")
    print(pairwise_results)
  } else {
    cat("Global test not significant. No pairwise comparisons performed.\n")
  }

  # Proportions
  proportions <- prop.table(contingency, margin = 2) * 100

  list(
    proportions = as.data.frame.matrix(proportions),
    chi2_pvalue = chi2_result$p.value,
    pairwise_tests = pairwise_results
  )
}


#' Statistical Test for Cluster Features
#'
#' Performs Mann-Whitney U tests between clusters for each feature.
#' Applies multiple comparison correction (Bonferroni or FDR).
#'
#' @param df_features Data frame with features and cluster assignments.
#'                    Last column assumed to be cluster labels.
#' @param clusters_col Character. Name of cluster column.
#' @param correction Character. Multiple comparison method ("bonferroni" or "fdr").
#'
#' @return List containing:
#'   \describe{
#'     \item{results}{Data frame with test results for all feature-cluster pairs}
#'     \item{significant_features}{Features with significant differences (p < 0.05)}
#'   }
#'
#' @details
#' For 2 clusters: performs pairwise Mann-Whitney U tests for each feature.
#' For >2 clusters: tests each cluster vs. rest.
#' Returns fold change as difference of medians.
#'
#' @examples
#' \dontrun{
#'   result <- statistical_test_clusters(df_with_clusters)
#'   sig_feats <- result$significant_features
#' }
#'
#' @export
statistical_test_clusters <- function(df_features,
                                       clusters_col = "Clusters",
                                       correction = "fdr") {
  # Extract clusters
  clusters <- df_features[[clusters_col]]
  unique_clusters <- sort(unique(clusters))
  n_clusters <- length(unique_clusters)

  # Feature columns (exclude cluster label)
  feature_cols <- colnames(df_features)[colnames(df_features) != clusters_col]

  all_results <- list()
  p_values <- c()

  if (n_clusters == 2) {
    # Pairwise test for 2 clusters
    cluster1_data <- df_features[clusters == unique_clusters[1], ]
    cluster2_data <- df_features[clusters == unique_clusters[2], ]

    for (feat in feature_cols) {
      test <- wilcox.test(cluster1_data[[feat]], cluster2_data[[feat]], exact = FALSE)

      fold_change <- median(cluster1_data[[feat]], na.rm = TRUE) -
        median(cluster2_data[[feat]], na.rm = TRUE)

      all_results[[length(all_results) + 1]] <- data.frame(
        Feature = feat,
        ClusterPair = sprintf("%s vs %s", unique_clusters[1], unique_clusters[2]),
        UStatistic = test$statistic,
        Pvalue = test$p.value,
        FoldChange = fold_change,
        stringsAsFactors = FALSE
      )

      p_values <- c(p_values, test$p.value)
    }
  } else {
    # Multiple clusters: test each vs rest
    for (i in seq_along(unique_clusters)) {
      cluster_id <- unique_clusters[i]
      cluster_data <- df_features[clusters == cluster_id, ]
      rest_data <- df_features[clusters != cluster_id, ]

      for (feat in feature_cols) {
        test <- wilcox.test(cluster_data[[feat]], rest_data[[feat]], exact = FALSE)

        fold_change <- median(cluster_data[[feat]], na.rm = TRUE) -
          median(rest_data[[feat]], na.rm = TRUE)

        all_results[[length(all_results) + 1]] <- data.frame(
          Feature = feat,
          ClusterPair = sprintf("%s vs Rest", cluster_id),
          UStatistic = test$statistic,
          Pvalue = test$p.value,
          FoldChange = fold_change,
          stringsAsFactors = FALSE
        )

        p_values <- c(p_values, test$p.value)
      }
    }
  }

  # Combine and adjust p-values
  results_df <- do.call(rbind, all_results)
  rownames(results_df) <- NULL

  if (correction == "fdr") {
    results_df$AdjustedPvalue <- p.adjust(results_df$Pvalue, method = "fdr")
  } else {
    results_df$AdjustedPvalue <- p.adjust(results_df$Pvalue, method = "bonferroni")
  }

  results_df$Significant <- results_df$AdjustedPvalue < 0.05

  significant_features <- results_df[results_df$Significant, ]

  list(
    results = results_df,
    significant_features = significant_features
  )
}


#' Feature Importance Analysis using Random Forest
#'
#' Trains Random Forest classifiers to compute feature importance for
#' distinguishing between clusters.
#'
#' @param df_features Data frame with features and cluster assignments.
#' @param clusters_col Character. Name of cluster column.
#' @param n_trees Integer. Number of trees in Random Forest (default 100).
#' @param seed Integer. Random seed (default 42).
#'
#' @return List containing:
#'   \describe{
#'     \item{importance_matrix}{Data frame with feature importances per cluster}
#'     \item{model_stats}{Summary of trained models}
#'   }
#'
#' @details
#' For 2 clusters: trains one binary classifier.
#' For >2 clusters: trains one classifier per cluster (vs. rest).
#'
#' @examples
#' \dontrun{
#'   result <- feature_importance_analysis(df_with_clusters)
#'   importance_mat <- result$importance_matrix
#' }
#'
#' @export
feature_importance_analysis <- function(df_features,
                                         clusters_col = "Clusters",
                                         n_trees = 100,
                                         seed = 42) {
  set.seed(seed)

  # Extract features and clusters
  X <- df_features[, colnames(df_features) != clusters_col]
  y <- df_features[[clusters_col]]

  unique_clusters <- sort(unique(y))
  n_clusters <- length(unique_clusters)

  feature_importances <- list()
  cluster_labels <- c()

  if (n_clusters == 2) {
    # Binary classification
    model <- randomForest::randomForest(X, factor(y), ntree = n_trees)
    feature_importances[[1]] <- model$importance[, 1]
    cluster_labels <- c(cluster_labels, sprintf("C%s", unique_clusters[1]))
    cluster_labels <- c(cluster_labels, sprintf("C%s", unique_clusters[2]))
    feature_importances[[2]] <- model$importance[, 1]
  } else {
    # One-vs-rest classification for each cluster
    for (i in seq_along(unique_clusters)) {
      cluster_id <- unique_clusters[i]
      y_binary <- ifelse(y == cluster_id, cluster_id, "Rest")
      model <- randomForest::randomForest(X, factor(y_binary), ntree = n_trees)
      feature_importances[[length(feature_importances) + 1]] <- model$importance[, 1]
      cluster_labels <- c(cluster_labels, sprintf("C%s", cluster_id))
    }
  }

  # Create importance matrix
  importance_matrix <- as.data.frame(do.call(rbind, feature_importances))
  colnames(importance_matrix) <- colnames(X)
  rownames(importance_matrix) <- cluster_labels
  importance_matrix <- as.matrix(importance_matrix)

  list(
    importance_matrix = importance_matrix,
    model_stats = list(
      n_clusters = n_clusters,
      n_features = ncol(X),
      n_samples = nrow(X)
    )
  )
}


#' Subpopulation Fingerprints
#'
#' Computes mean feature values for each subpopulation to create
#' a fingerprint profile.
#'
#' @param df_features Data frame with features and cluster assignments.
#' @param clusters_col Character. Name of cluster column.
#'
#' @return List containing:
#'   \describe{
#'     \item{fingerprint_matrix}{Mean feature values per cluster}
#'     \item{summary_stats}{Summary statistics per cluster}
#'   }
#'
#' @examples
#' \dontrun{
#'   result <- subpopulation_fingerprints(df_with_clusters)
#'   fingerprint_mat <- result$fingerprint_matrix
#' }
#'
#' @export
subpopulation_fingerprints <- function(df_features,
                                        clusters_col = "Clusters") {
  # Extract features
  feature_cols <- colnames(df_features)[colnames(df_features) != clusters_col]
  clusters <- df_features[[clusters_col]]

  # Compute mean per cluster
  fingerprint_list <- list()

  for (cluster_id in sort(unique(clusters))) {
    cluster_data <- df_features[clusters == cluster_id, feature_cols]
    means <- colMeans(cluster_data, na.rm = TRUE)
    fingerprint_list[[as.character(cluster_id)]] <- means
  }

  fingerprint_matrix <- do.call(rbind, fingerprint_list)

  list(
    fingerprint_matrix = fingerprint_matrix,
    summary_stats = list(
      n_clusters = nrow(fingerprint_matrix),
      n_features = ncol(fingerprint_matrix),
      mean_values = colMeans(fingerprint_matrix, na.rm = TRUE)
    )
  )
}
