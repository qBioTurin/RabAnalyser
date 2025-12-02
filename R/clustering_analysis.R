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
  # Removes highly correlated features from a DataFrame,
  # retaining the one with higher variance.

  if("Class" %in% colnames(df)){
    treatment_label <- df %>% select(Class) %>% pull()
    df <- df %>% select(-Class)
  }else{
    treatment_label = NULL
  }

  # Compute correlation matrix
  cor_mat <- cor(df, use = "pairwise.complete.obs")
  feats   <- colnames(df)
  n_feats <- length(feats)

  # Find feature pairs with correlation above the threshold (i < j, avoid duplicates)
  correlated_pairs <- data.frame(
    feature1 = character(),
    feature2 = character(),
    abs_corr = numeric(),
    stringsAsFactors = FALSE
  )

  for (i in seq_len(n_feats)) {
    for (j in seq_len(n_feats)) {
      if (i < j) {
        c_ij <- cor_mat[i, j]
        if (!is.na(c_ij) && abs(c_ij) > threshold) {
          correlated_pairs <- rbind(
            correlated_pairs,
            data.frame(
              feature1 = feats[i],
              feature2 = feats[j],
              abs_corr = abs(c_ij),
              stringsAsFactors = FALSE
            )
          )
        }
      }
    }
  }

  if (nrow(correlated_pairs) != 0) {
  # Sort correlated pairs by descending correlation
  correlated_pairs <- correlated_pairs[order(-correlated_pairs$abs_corr), ]

  removed_features <- character(0)  # set of removed features

  # Remove features with lower variance in each correlated pair
  for (row_idx in seq_len(nrow(correlated_pairs))) {
    feature1 <- correlated_pairs$feature1[row_idx]
    feature2 <- correlated_pairs$feature2[row_idx]

    # Skip if already removed
    if (feature1 %in% removed_features || feature2 %in% removed_features) {
      next
    }

    # Compare variances
    v1 <- var(df[[feature1]], na.rm = TRUE)
    v2 <- var(df[[feature2]], na.rm = TRUE)

    if (v1 > v2) {
      removed_features <- c(removed_features, feature2)
    } else {
      removed_features <- c(removed_features, feature1)
    }
  }


  # Return DataFrame with redundant features removed
  df = df[, !(colnames(df) %in% removed_features), drop = FALSE]
  }

  # Filtered correlation matrix
  corr_filtered <- cor(df, use = "pairwise.complete.obs")

  if(!is.null(treatment_label)){
    df$Class = treatment_label
  }

  list(
    filtered_data = df,
    corr_original = cor_mat,
    corr_filtered = corr_filtered
  )

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

#' Cluster-wise feature statistics using Mann-Whitney tests
#'
#' This function performs statistical comparisons of feature distributions
#' between clusters and summarizes the results in an Excel file and dot plots.
#' For each feature, pairwise Mann-Whitney (Wilcoxon rank-sum) tests are
#' computed between clusters.
#'
#' When there are only two clusters, a single Mann-Whitney test is performed
#' per feature without multiple-testing correction. When there are more than
#' two clusters, all pairwise comparisons are performed and p-values are
#' adjusted using Bonferroni correction.
#'
#' The function also generates a dot plot:
#' \itemize{
#'   \item Color encodes the fold change (difference of medians) between clusters.
#'   \item Point size encodes the statistical significance (-log10(p-value)).
#' }
#'
#' @param reduced_df A \code{data.frame} containing numeric features and
#'   cluster assignments. By default, the function assumes that the last two
#'   columns are \code{Clusters} and \code{Conditions}, and all preceding
#'   columns are numeric features.
#'
#' @return Invisibly returns a \code{data.frame} containing all statistical
#'   results (one row per feature and pair of clusters). The Excel file is
#'   written to disk and the dot plots are printed to the active graphics device.
#'
#' @details
#' The function internally:
#' \enumerate{
#'   \item Determines the number of clusters from the \code{Clusters} column
#'   \item Uses all columns except the last two in \code{reduced_df} as
#'         features.
#'   \item For 2 clusters: performs one Mann-Whitney test per feature and
#'         reports p-values and fold changes.
#'   \item For more than 2 clusters: performs all pairwise Mann-Whitney tests
#'         per feature, applies Bonferroni correction, and indicates which
#'         comparisons are significant.
#'   \item Produces a dot plot summarizing the results:
#'         \itemize{
#'           \item Two clusters: size = -log10(raw p-value), color = fold change.
#'           \item More than two clusters: size = -log10(corrected p-value),
#'                 color = fold change.
#'         }
#' }
#'
#' @import rstatix
#' @import ggplot2
#' @import dplyr
#' @export

cluster_feature_stats <- function(reduced_df) {

  lapply(colnames(reduced_df %>% select(-Condition, -Clusters)), function(feature) {
    #### STATISTICAL TEST ####
    # Perform pairwise Wilcoxon test (Mann-Whitney U test)
    test_results <- reduced_df %>%
      rstatix::wilcox_test(as.formula(paste(feature, "~ Clusters")), alternative = "two.sided", exact = FALSE) %>%
      rstatix::adjust_pvalue(method = "bonferroni") %>% # Adjust for multiple testing
      mutate(Significant = ifelse(p.adj < 0.05, "*", "ns"))  # Mark significance

    # Compute median differences for each pair
    cluster_medians <- reduced_df %>%
      mutate(Clusters = as.character(Clusters)) %>%
      group_by(Clusters) %>%
      summarise(median_val = median(!!sym(feature)) )

    test_results <- test_results %>%
      left_join(cluster_medians, by = c("group1" = "Clusters")) %>%
      rename(median1 = median_val) %>%
      left_join(cluster_medians, by = c("group2" = "Clusters")) %>%
      rename(median2 = median_val) %>%
      mutate(Fold_Change = median1 - median2)

    SM_final <- test_results %>%
      mutate(Size = ifelse(p.adj <= 0.05, pmin(-log10(p.adj), 20), NA))

    SM_final$Color <- scales::col_numeric(
      palette = colorRampPalette(c("#997FD2", "#F3A341"))(100),
      domain = range(SM_final$Fold_Change, na.rm = TRUE)
    )(SM_final$Fold_Change)
    SM_final$Feature = feature

    SM_final$Cluster_pairs = paste0(SM_final$group1," vs ",SM_final$group2)
    SM_final
  }) -> StatList

  SM_final = do.call(rbind, StatList)
  SM_final$neg_log10_p <- -log10(SM_final$p.adj)
  SM_final$Significant <- SM_final$p.adj < 0.05

  vmin <- min(SM_final$Fold_Change, na.rm = TRUE)
  vmax <- max(SM_final$Fold_Change, na.rm = TRUE)

  size_legend_values <- c(1.5, 2.5, 5, 7.5, 10)

  # Keep only significant comparisons for plotting
  df_significant <- dplyr::filter(SM_final, Significant)

    # Dot plot for multiple clusters
    p <- ggplot(df_significant,
                aes(x = Cluster_pairs, y = Feature)) +
      geom_point(
        aes(size = neg_log10_p, fill = Fold_Change),
        shape = 21, color = "black", alpha = 0.75
      ) +
      scale_size_continuous(
        range = c(3, 20),
        breaks = size_legend_values,
        name = "-log10(p)"
      ) +
      scale_fill_gradient2(
        low = "blue", mid = "white", high = "green",
        midpoint = 0,
        limits = c(vmin, vmax),
        name = "Median difference"
      ) +
      labs(
        x = "Cluster pairs",
        y = "Feature",
        title = "Dot plot of Features vs Cluster Pairs"
      ) +
      theme_minimal() +
      theme(
        axis.text.x = element_text(angle = 45, hjust = 1),
        legend.position = "right"
      )

  # Return the full statistics table invisibly
  return( list(SM_final = SM_final, plot =p) )
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
#' @import tidyr
#' @import dplyr
#' @import ggplot2
#' @import randomForest
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
  CLsums = apply(importance_matrix,1,sum)
  for(i in 1:nrow(importance_matrix)) importance_matrix[i,] = importance_matrix[i,]/CLsums[i]

  colnames(importance_matrix) <- colnames(X)
  rownames(importance_matrix) <- cluster_labels

  importance_df = importance_matrix %>%
    dplyr::mutate(Clusters = gsub(pattern = "C",replacement = "",x = row.names(importance_matrix))) %>%
    tidyr::gather(-Clusters, key = "Feature", value = "Importance")

  importance_matrix <- as.matrix(importance_matrix)

  ImportancePLot = ggplot(importance_df, aes(x = Feature, y = Clusters, fill = Importance)) +
    geom_tile(color = "white") +
    geom_text(aes(label = sprintf("%.2f", Importance)),   # format with 2 decimals
              color = "black", size = 3) +
    scale_fill_gradient(low = "white", high = "black",limits = c(0,1)) +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1),
          legend.position = "right") +
    labs(title = "Feature Importance Heatmap",
         x = "Feature",
         y = "Cluster",
         fill = "Importance")


  list(
    importance_matrix = importance_matrix,
    plot = ImportancePLot,
    model_stats = list(
      n_clusters = n_clusters,
      n_features = ncol(X),
      n_samples = nrow(X)
    )
  )
}




