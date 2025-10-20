#' Visualize Features on UMAP
#'
#' Creates UMAP scatter plots colored by individual feature values.
#' Saves plots as PNG and SVG for each feature.
#'
#' @param umap_coords Data frame with UMAP1 and UMAP2 coordinates.
#' @param df_features Data frame with feature values.
#' @param output_dir Character. Directory where to save plots.
#' @param vmin Numeric. Minimum color scale value (default -0.4).
#' @param vmax Numeric. Maximum color scale value (default 0.4).
#' @param save_plots Logical. Whether to save plots to disk (default TRUE).
#' @param cmap Character. Color map name (default "RdBu").
#'
#' @return Invisible list of ggplot objects.
#'
#' @examples
#' \dontrun{
#'   visualize_feature_values_umap(umap_coords, df_features, output_dir = ".")
#' }
#'
#' @import ggplot2
#' @export
visualize_feature_values_umap <- function(umap_coords, df_features,
                                           output_dir = ".",
                                           vmin = -0.4, vmax = 0.4,
                                           save_plots = TRUE,
                                           cmap = "RdBu") {
  # Get feature columns
  feature_cols <- colnames(df_features)

  plots <- list()

  for (feat in feature_cols) {
    # Create data for plot
    plot_data <- cbind(
      umap_coords,
      Feature = df_features[[feat]]
    )

    p <- ggplot2::ggplot(plot_data, ggplot2::aes(x = .data$UMAP1, y = .data$UMAP2, color = .data$Feature)) +
      ggplot2::geom_point(alpha = 0.8, size = 2) +
      ggplot2::scale_color_distiller(palette = cmap, limits = c(vmin, vmax)) +
      ggplot2::labs(
        title = sprintf("UMAP: %s", feat),
        x = "UMAP1",
        y = "UMAP2",
        color = feat
      ) +
      ggplot2::theme_minimal() +
      ggplot2::theme(
        plot.title = ggplot2::element_text(hjust = 0.5, size = 12),
        axis.title = ggplot2::element_text(size = 10)
      )

    plots[[feat]] <- p

    if (save_plots) {
      # Sanitize filename
      safe_feat <- gsub("[^[:alnum:]._-]", "_", feat)

      png_path <- file.path(output_dir, sprintf("UMAP_visual_%s.png", safe_feat))
      svg_path <- file.path(output_dir, sprintf("UMAP_visual_%s.svg", safe_feat))

      ggplot2::ggsave(png_path, p, width = 8, height = 6, dpi = 300)
      ggplot2::ggsave(svg_path, p, width = 8, height = 6)

      cat(sprintf("Saved: %s, %s\n", png_path, svg_path))
    }
  }

  invisible(plots)
}


#' Plot UMAP with Leiden Clusters
#'
#' Creates scatter plot of UMAP coordinates colored by cluster assignments.
#'
#' @param umap_coords Data frame with UMAP1 and UMAP2 coordinates.
#' @param clusters Integer vector of cluster assignments.
#' @param title Character. Plot title.
#' @param save_path Character or NULL. If provided, saves plot to this path.
#'
#' @return ggplot object.
#'
#' @examples
#' \dontrun{
#'   plot_umap_clusters(umap_coords, clusters, save_path = "clusters.png")
#' }
#'
#' @import ggplot2
#' @export
plot_umap_clusters <- function(umap_coords, clusters,
                                title = "Leiden Clustering in UMAP Space",
                                save_path = NULL) {
  plot_data <- cbind(umap_coords, Cluster = factor(clusters))

  p <- ggplot2::ggplot(plot_data, ggplot2::aes(x = .data$UMAP1, y = .data$UMAP2, color = .data$Cluster)) +
    ggplot2::geom_point(alpha = 0.7, size = 2) +
    ggplot2::scale_color_viridis_d() +
    ggplot2::labs(
      title = title,
      x = "UMAP1",
      y = "UMAP2"
    ) +
    ggplot2::theme_minimal() +
    ggplot2::theme(
      plot.title = ggplot2::element_text(hjust = 0.5, size = 12),
      legend.position = "right"
    )

  if (!is.null(save_path)) {
    ggplot2::ggsave(save_path, p, width = 8, height = 6)
    cat(sprintf("Saved: %s\n", save_path))
  }

  p
}


#' Plot UMAP by Condition
#'
#' Creates scatter plot of UMAP coordinates with points colored by condition.
#'
#' @param umap_coords Data frame with UMAP1 and UMAP2 coordinates.
#' @param conditions Character vector of condition labels.
#' @param title Character. Plot title.
#'
#' @return ggplot object.
#'
#' @examples
#' \dontrun{
#'   plot_umap_conditions(umap_coords, conditions)
#' }
#'
#' @import ggplot2
#' @export
plot_umap_conditions <- function(umap_coords, conditions,
                                  title = "UMAP Visualization by Condition") {
  plot_data <- cbind(umap_coords, Condition = factor(conditions))

  p <- ggplot2::ggplot(plot_data, ggplot2::aes(x = .data$UMAP1, y = .data$UMAP2, color = .data$Condition)) +
    ggplot2::geom_point(alpha = 0.7, size = 2) +
    ggplot2::scale_color_brewer(palette = "Set2") +
    ggplot2::labs(
      title = title,
      x = "UMAP1",
      y = "UMAP2"
    ) +
    ggplot2::theme_minimal() +
    ggplot2::theme(
      plot.title = ggplot2::element_text(hjust = 0.5, size = 12),
      legend.position = "right"
    )

  p
}


#' Plot Correlation Matrices
#'
#' Creates heatmaps of feature correlation matrices.
#'
#' @param corr_matrix Matrix. Correlation matrix to plot.
#' @param title Character. Plot title.
#' @param save_path Character or NULL. If provided, saves plot.
#'
#' @return ggplot object (from pheatmap).
#'
#' @examples
#' \dontrun{
#'   plot_correlation_matrix(corr_original, "Correlation Matrix - Original Features")
#' }
#'
#' @export
plot_correlation_matrix <- function(corr_matrix, title = "Correlation Matrix",
                                     save_path = NULL) {
  p <- pheatmap::pheatmap(
    corr_matrix,
    main = title,
    color = colorRampPalette(c("blue", "white", "red"))(50),
    breaks = seq(-1, 1, length.out = 51),
    display_numbers = TRUE,
    number_format = "%.2f",
    fontsize_number = 8,
    cluster_rows = TRUE,
    cluster_cols = TRUE
  )

  if (!is.null(save_path)) {
    ggplot2::ggsave(save_path, p, width = 10, height = 8)
    cat(sprintf("Saved: %s\n", save_path))
  }

  p
}


#' Plot Leiden Clustering Metrics
#'
#' Creates plots of silhouette, DBI, and modularity scores vs. resolution.
#'
#' @param leiden_result List from leiden_clustering_optimization().
#' @param output_dir Character. Directory to save plots.
#'
#' @return Invisible list of ggplot objects.
#'
#' @import ggplot2
#' @export
plot_leiden_metrics <- function(leiden_result, output_dir = ".") {
  resolutions <- leiden_result$resolutions
  silhouette <- leiden_result$silhouette_scores
  dbi <- leiden_result$dbi_scores
  modularity <- leiden_result$modularity_scores
  n_clusters <- leiden_result$num_clusters

  # Create data frame
  metrics_df <- data.frame(
    Resolution = resolutions,
    Silhouette = silhouette,
    DBI = dbi,
    Modularity = modularity,
    NClusters = n_clusters
  )

  # Plot 1: Silhouette & DBI
  p1 <- ggplot2::ggplot(metrics_df) +
    ggplot2::geom_line(ggplot2::aes(x = .data$Resolution, y = .data$Silhouette), color = "blue", linewidth = 0.8) +
    ggplot2::geom_point(ggplot2::aes(x = .data$Resolution, y = .data$Silhouette), color = "blue", size = 1) +
    ggplot2::geom_line(ggplot2::aes(x = .data$Resolution, y = .data$DBI), color = "red", linewidth = 0.8) +
    ggplot2::geom_point(ggplot2::aes(x = .data$Resolution, y = .data$DBI), color = "red", size = 1) +
    ggplot2::labs(
      title = "Silhouette Score & Davies-Bouldin Index",
      x = "Resolution",
      y = "Score"
    ) +
    ggplot2::theme_minimal()

  # Plot 2: Modularity
  p2 <- ggplot2::ggplot(metrics_df, ggplot2::aes(x = .data$Resolution, y = .data$Modularity)) +
    ggplot2::geom_line(color = "darkred", linewidth = 0.8) +
    ggplot2::geom_point(color = "darkred", size = 1) +
    ggplot2::labs(
      title = "Modularity Score",
      x = "Resolution",
      y = "Modularity"
    ) +
    ggplot2::theme_minimal()

  # Plot 3: Number of clusters
  p3 <- ggplot2::ggplot(metrics_df, ggplot2::aes(x = .data$Resolution, y = .data$NClusters)) +
    ggplot2::geom_line(color = "darkgreen", linewidth = 0.8) +
    ggplot2::geom_point(color = "darkgreen", size = 1) +
    ggplot2::labs(
      title = "Number of Clusters",
      x = "Resolution",
      y = "Clusters"
    ) +
    ggplot2::theme_minimal()

  # Save
  ggplot2::ggsave(file.path(output_dir, "Leiden_Silhouette_DBI.png"), p1, width = 10, height = 6)
  ggplot2::ggsave(file.path(output_dir, "Leiden_Modularity.png"), p2, width = 10, height = 6)
  ggplot2::ggsave(file.path(output_dir, "Leiden_NClusters.png"), p3, width = 10, height = 6)

  invisible(list(silhouette_dbi = p1, modularity = p2, n_clusters = p3))
}


#' Plot Feature Importance Heatmap
#'
#' Creates heatmap of Random Forest feature importances.
#'
#' @param importance_matrix Matrix. Feature importance values (features x clusters).
#' @param title Character. Plot title.
#' @param save_path Character or NULL. If provided, saves plot.
#'
#' @return ggplot object (from pheatmap).
#'
#' @export
plot_feature_importance <- function(importance_matrix,
                                     title = "Feature Importance Heatmap",
                                     save_path = NULL) {
  p <- pheatmap::pheatmap(
    importance_matrix,
    main = title,
    color = colorRampPalette(c("white", "gray20"))(50),
    display_numbers = TRUE,
    number_format = "%.3f",
    fontsize_number = 8,
    cluster_rows = TRUE,
    cluster_cols = FALSE,
    cellwidth = 20,
    cellheight = 15
  )

  if (!is.null(save_path)) {
    ggplot2::ggsave(save_path, p, width = 12, height = 8)
    cat(sprintf("Saved: %s\n", save_path))
  }

  p
}


#' Plot Subpopulation Proportions
#'
#' Creates stacked bar plot of cell proportions across conditions.
#'
#' @param proportions_df Data frame with proportions (conditions x clusters).
#' @param title Character. Plot title.
#' @param save_path Character or NULL. If provided, saves plot.
#'
#' @return ggplot object.
#'
#' @import ggplot2
#' @export
plot_subpopulation_proportions <- function(proportions_df,
                                            title = "Cell Proportions by Condition",
                                            save_path = NULL) {
  # Reshape to long format
  plot_data <- tidyr::pivot_longer(
    as.data.frame(proportions_df),
    cols = everything(),
    names_to = "Condition",
    values_to = "Proportion"
  )

  plot_data$Cluster <- rep(rownames(proportions_df), ncol(proportions_df))

  p <- ggplot2::ggplot(plot_data, ggplot2::aes(x = .data$Condition, y = .data$Proportion, fill = .data$Cluster)) +
    ggplot2::geom_bar(stat = "identity", position = "stack") +
    ggplot2::scale_fill_viridis_d() +
    ggplot2::labs(
      title = title,
      x = "Condition",
      y = "Proportion (%)",
      fill = "Cluster"
    ) +
    ggplot2::theme_minimal() +
    ggplot2::theme(
      plot.title = ggplot2::element_text(hjust = 0.5),
      axis.text.x = ggplot2::element_text(angle = 45, hjust = 1)
    )

  if (!is.null(save_path)) {
    ggplot2::ggsave(save_path, p, width = 8, height = 6)
    cat(sprintf("Saved: %s\n", save_path))
  }

  p
}


#' Plot Fingerprint Heatmap
#'
#' Creates heatmap of mean feature values per subpopulation.
#'
#' @param fingerprint_matrix Matrix. Mean feature values (clusters x features).
#' @param title Character. Plot title.
#' @param save_path Character or NULL. If provided, saves plot.
#'
#' @return ggplot object (from pheatmap).
#'
#' @export
plot_fingerprints <- function(fingerprint_matrix,
                               title = "Subpopulation Fingerprints",
                               save_path = NULL) {
  p <- pheatmap::pheatmap(
    fingerprint_matrix,
    main = title,
    color = colorRampPalette(c("blue", "white", "red"))(50),
    display_numbers = TRUE,
    number_format = "%.2f",
    fontsize_number = 8,
    cluster_rows = FALSE,
    cluster_cols = TRUE,
    scale = "column"  # Scale by column (feature)
  )

  if (!is.null(save_path)) {
    ggplot2::ggsave(save_path, p, width = 12, height = 6)
    cat(sprintf("Saved: %s\n", save_path))
  }

  p
}


#' Plot Statistical Test Results as Dot Plot
#'
#' Visualizes Mann-Whitney U test results as dot plot with
#' size and color encoding significance and effect size.
#'
#' @param stats_results Data frame from statistical_test_clusters().
#' @param title Character. Plot title.
#' @param save_path Character or NULL. If provided, saves plot.
#'
#' @return ggplot object.
#'
#' @import ggplot2
#' @export
plot_statistical_results <- function(stats_results,
                                      title = "Statistical Test Results",
                                      save_path = NULL) {
  # Add -log10(p-value) for size
  stats_results$LogPvalue <- -log10(stats_results$AdjustedPvalue)
  stats_results$LogPvalue[is.infinite(stats_results$LogPvalue)] <- 20  # Cap extreme values

  # Limit to significant results for cleaner plot
  stats_sig <- stats_results[stats_results$Significant, ]

  if (nrow(stats_sig) == 0) {
    cat("No significant results to plot\n")
    return(NULL)
  }

  p <- ggplot2::ggplot(stats_sig, ggplot2::aes(
    x = .data$ClusterPair, y = .data$Feature,
    size = .data$LogPvalue, color = .data$FoldChange
  )) +
    ggplot2::geom_point(alpha = 0.7) +
    ggplot2::scale_color_gradient2(
      low = "blue", mid = "white", high = "red",
      midpoint = 0
    ) +
    ggplot2::scale_size_continuous(
      name = "-log10(p-value)",
      range = c(2, 8)
    ) +
    ggplot2::labs(
      title = title,
      x = "Cluster Pair",
      y = "Feature",
      color = "Fold Change"
    ) +
    ggplot2::theme_minimal() +
    ggplot2::theme(
      plot.title = ggplot2::element_text(hjust = 0.5),
      axis.text.x = ggplot2::element_text(angle = 45, hjust = 1)
    )

  if (!is.null(save_path)) {
    ggplot2::ggsave(save_path, p, width = 10, height = 8)
    cat(sprintf("Saved: %s\n", save_path))
  }

  p
}
