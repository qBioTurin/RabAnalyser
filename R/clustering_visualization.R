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
#' @import ggplot2 reshape2
#'
plot_correlation_matrix <- function(corr_matrix, title = "Correlation Matrix",
                                     save_path = NULL) {

  # corrplot(corr_matrix, method = "color", type = "full",addCoef.col = 'black', col = colorRampPalette(c("blue", "white", "red"))(200),
  #          tl.cex = 0.6, number.cex = 0.5, title = "Correlation Matrix of original Features",
  #          mar = c(0, 0, 2, 0))

  # Convert matrix to long format
  corr_df <- melt(corr_matrix)
  colnames(corr_df) <- c("Var1", "Var2", "Correlation")

  # Create ggplot heatmap
  p <- ggplot(corr_df, aes(x = Var1, y = Var2, fill = Correlation)) +
    geom_tile(color = "white") +
    scale_fill_gradient2(
      low = "blue", mid = "white", high = "red",
      midpoint = 0, limits = c(-1, 1)
    ) +
    geom_text(aes(label = sprintf("%.2f", Correlation)),
              color = "black", size = 2, fontface = "bold" ) +
    labs(title = title, x = "", y = "") +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5),
          axis.text.y = element_text(size = 8),
          plot.title = element_text(hjust = 0.5, size = 14, face = "bold"))

  # Save if path provided
  if (!is.null(save_path)) {
    ggsave(save_path, p, width = 10, height = 8)
    cat(sprintf("Saved: %s\n", save_path))
  }

  return(p)
}

#' Plot UMAP with flexible coloring (discrete or continuous)
#'
#' @param umap_df Data frame from run_umap_leiden()$umap_df
#' @param color_by Character. Column name to color points by. Default "Class".
#' @param point_size Numeric. Size of points. Default 1.
#' @param discrete Logical or NULL.
#' @param low_color Color for minimum (continuous). Default "blue".
#' @param mid_color Color for midpoint (continuous). Default "white".
#' @param high_color Color for maximum (continuous). Default "green".
#' @param values_interval Numeric vector, lower and upper bounds of the color scale (default NULL).
#' @param midpoint_val Numeric, midpoint of the color scale (default NULL).
#'
#' @export
#' @import ggplot2
#' @importFrom rlang .data
plot_umap <- function(
    umap_df,
    color_by   = "Class",
    point_size = 1,
    discrete   = NULL,
    low_color  = "blue",
    mid_color  = "white",
    high_color = "green",
    values_interval = NULL,
    midpoint_val = NULL
) {
  # Controllo colonna
  if (!color_by %in% names(umap_df)) {
    stop(sprintf("Column '%s' not found in umap_df.", color_by))
  }

  col_data <- umap_df[[color_by]]

  if (is.null(discrete)) {
    discrete <- !is.numeric(col_data)
  }

  p <- ggplot2::ggplot(
    umap_df,
    ggplot2::aes(x = UMAP1, y = UMAP2)
  )

  if (discrete) {
    p <- p +
      ggplot2::geom_point(
        ggplot2::aes(color = factor(.data[[color_by]])),
        size = point_size
      ) +
      ggplot2::scale_color_discrete(name = color_by)

  } else {

    if(is.null(midpoint_val)){
      midpoint_val <- stats::median(col_data, na.rm = TRUE)
    }

    if(!is.null(values_interval)){
      col_data = scales::rescale(col_data,c(min(values_interval),max(values_interval)) )
      values_interval = c(min(values_interval),midpoint_val,max(values_interval))
    }else {
      values_interval = c(min(col_data),midpoint_val,max(col_data))
    }

    p <- p +
      ggplot2::geom_point(
        ggplot2::aes(color = .data[[color_by]]),
        size = point_size
      ) +
      ggplot2::scale_color_gradient2(
        low      = low_color,
        mid      = mid_color,
        high     = high_color,
        midpoint = midpoint_val,
        name     = color_by,
        breaks=values_interval,
        labels=values_interval,
        limits=c(min(values_interval),max(values_interval))
      )
  }

  p +
    ggplot2::theme_minimal() +
    ggplot2::labs(color = color_by) +
    ggplot2::theme(
      plot.title      = ggplot2::element_text(hjust = 0.5, face = "bold"),
      legend.position = "right"
    )
}


#' Plot resolution scan
#'
#' @param resolution_scan Data frame from run_umap_leiden()$resolution_scan
#' @export
#' @import ggplot2
plot_resolution_scan <- function(resolution_scan) {

  ggplot2::ggplot(resolution_scan, ggplot2::aes(x = gamma, y = n_clusters)) +
    ggplot2::geom_line() +
    ggplot2::geom_point(size = 2) +
    ggplot2::theme_minimal() +
    ggplot2::labs(
      x = "Resolution Parameter (γ)",
      y = "Number of Clusters",
      title = "Leiden Clustering: Resolution vs Number of Clusters"
    ) +
    ggplot2::theme(plot.title = ggplot2::element_text(hjust = 0.5, face = "bold"))+
    ggplot2::scale_y_continuous(breaks = 1:max(resolution_scan$n_clusters),
                                labels = 1:max(resolution_scan$n_clusters))
}


#' Plot cluster stability
#'
#' @param stability Data frame from run_umap_leiden()$stability
#' @param threshold Numeric. Horizontal line marking stability threshold. Default 0.6.
#' @export
#' @import ggplot2
plot_cluster_stability <- function(stability, threshold = 0.6) {

  ggplot2::ggplot(stability, ggplot2::aes(x = factor(cluster), y = jaccard)) +
    ggplot2::geom_boxplot(fill = "lightblue", color = "black", outlier.shape = 19) +
    ggplot2::geom_hline(yintercept = threshold, linetype = "dashed", color = "gray") +
    ggplot2::theme_minimal() +
    ggplot2::labs(
      x = "Cluster Index",
      y = "Jaccard Index",
      title = "Cluster Stability Across Bootstrap Runs"
    ) +
    ggplot2::theme(
      plot.title = ggplot2::element_text(hjust = 0.5, face = "bold")
    )
}



#' Plot Clusters Proportions
#'
#' Creates stacked bar plot of cell proportions across conditions.
#'
#' @param reduced_df  A \code{data.frame} containing numeric features and
#'   cluster assignments. By default, the function assumes that the last two
#'   columns are \code{Clusters} and \code{Conditions}, and all preceding
#'   columns are numeric features.
#' @param title Character. Plot title.
#' @param save_path Character or NULL. If provided, saves plot.
#'
#' @return ggplot object.
#'
#' @import ggplot2
#' @export
plot_clusters_proportions <- function(reduced_df,
                                           title = "Cell Proportions by Condition",
                                           save_path = NULL) {

  total_cells_per_condition <- table(reduced_df$Condition)

  cluster_counts <- reduced_df %>%
    dplyr::count(Clusters, Condition, name = "Count") %>%
    tidyr::pivot_wider(
      names_from  = Condition,
      values_from = Count,
      values_fill = 0
    )

  condition_names <- setdiff(colnames(cluster_counts), "Clusters")
  cluster_fractions <- cluster_counts

  for (cond in condition_names) {
    cluster_fractions[[cond]] <- (cluster_fractions[[cond]] /
                                    as.numeric(total_cells_per_condition[cond])) * 100
  }

  frac_long <- cluster_fractions %>%
    tidyr::pivot_longer(
      cols = all_of(condition_names),
      names_to = "Condition",
      values_to = "Fraction"
    )

  p <- ggplot(frac_long, aes(x = Clusters, y = Fraction, fill = Condition)) +
    geom_col(position = position_dodge(width = 0.8), width = 0.7) +
    scale_y_continuous(limits = c(0, 100)) +
    labs(
      x = "Clusters",
      y = "Cell proportion (%)",
      title = "Comparison of Cell Fractions Across Conditions Per Cluster"
    ) +
    theme_minimal() +
    theme(
      axis.text.x = element_text(angle = 45, hjust = 1)
    )

  if (!is.null(save_path)) {
    ggplot2::ggsave(save_path, p, width = 8, height = 6)
    cat(sprintf("Saved: %s\n", save_path))
  }

  p
}


#' Cluster fingerprint heatmap with custom white-band colormap
#'
#' This function computes a "fingerprint" matrix by averaging feature values
#' per cluster and visualizes it as a heatmap with a custom diverging colormap
#' that includes a white band around zero.
#'
#' The intended use case is for Kolmogorov-Smirnov (KS) statistics or similar
#' effect-size measures, where:
#' \itemize{
#'   \item KS < 0.15: small effect
#'   \item KS 0.15–0.25: moderate effect
#'   \item KS > 0.3: strong effect
#' }
#' These thresholds are arbitrary and can be chosen by visual inspection of
#' the distribution shifts.
#'
#' @param reduced_df A \code{data.frame} where:
#'   \itemize{
#'     \item All columns except the last two are numeric features.
#'     \item One of the last two columns is a cluster identifier named
#'           \code{"Clusters"} (e.g., cluster labels or subpopulation IDs).
#'     \item The other last column may be a grouping variable (e.g.,
#'           \code{"conditions"}); it is ignored here but kept for consistency
#'           with the original Python code.
#'   }
#' @param values_interval Numeric vector, lower and upper bounds of the color scale (default NULL).
#' @param midpoint_val Numeric, midpoint of the color scale (default NULL).
#' @return A list with:
#'   \itemize{
#'     \item \code{fingerprint_df}: a \code{data.frame} of mean feature values
#'           per cluster (rows = clusters, columns = features).
#'     \item \code{plot}: the \code{ggplot2} object for the heatmap.
#'   }
#'   The function also prints the heatmap to the current graphics device.
#'
#' @details
#' Internally, the function:
#' \enumerate{
#'   \item Extracts feature columns as all columns except the last two.
#'   \item Groups rows by \code{Clusters} and computes the mean of each feature.
#'   \item Builds a custom diverging palette with:
#'         \itemize{
#'           \item A gradient from \code{vmin} up to \code{white_min} using
#'                 \code{left_colors}.
#'           \item A white band from \code{white_min} to \code{white_max}.
#'           \item A gradient from \code{white_max} to \code{vmax} using
#'                 \code{right_colors}.
#'         }
#'   \item Plots a heatmap using \code{ggplot2::geom_tile} with cluster
#'         on the y-axis, feature on the x-axis, and color encoding the
#'         mean value.
#' }
#'
#' @import ggplot2
#' @importFrom tidyr pivot_longer
#' @importFrom tibble rownames_to_column
#' @importFrom dplyr group_by summarise across everything
#' @importFrom scales squish
#'
#' @export
ks_cluster_fingerprint_heatmap <- function(reduced_df,
                                           values_interval = NULL,
                                           midpoint_val = NULL,
                                           low_color  = c("#2166ac"),
                                           mid_color  = c("white"),
                                           high_color = c( "#1b7837"),
                                           n_colors = 512) {

  # ------------------------------------------------------------------
  # 1. Extract features and compute mean per cluster ("fingerprint")
  # ------------------------------------------------------------------

  # Assume: all columns except the last two are numeric features
  if (ncol(reduced_df) < 3) {
    stop("Expected at least 3 columns: features + Clusters + another column.")
  }

  feature_cols <- colnames(reduced_df)[1:(ncol(reduced_df) - 2)]

  if (!"Clusters" %in% colnames(reduced_df)) {
    stop("Column 'Clusters' not found in 'reduced_df'.")
  }

  # Compute mean feature values per cluster
  fingerprint_df <- reduced_df |>
    dplyr::group_by(.data$Clusters) |>
    dplyr::summarise(
      dplyr::across(all_of(feature_cols), ~ mean(.x, na.rm = TRUE)),
      .groups = "drop"
    )

  # Set rownames to cluster labels and drop 'Clusters' column from data matrix
  fingerprint_mat <- as.data.frame(fingerprint_df)
  rownames(fingerprint_mat) <- fingerprint_mat$Clusters
  fingerprint_mat$Clusters <- NULL

  # ------------------------------------------------------------------
  # 3. Reshape to long format for ggplot
  # ------------------------------------------------------------------
  plot_df <- fingerprint_mat |>
    tibble::rownames_to_column(var = "Cluster") |>
    tidyr::pivot_longer(
      cols = -Cluster,
      names_to = "Feature",
      values_to = "Value"
    )

  # Ensure ordered clusters if needed
  plot_df$Cluster <- factor(plot_df$Cluster, levels = sort(unique(plot_df$Cluster)))

  if(!is.null(values_interval)){
    plot_df$Value = scales::rescale(plot_df$Value,c(min(values_interval),max(values_interval)) )
  }

  if(is.null(midpoint_val)){
    midpoint_val <- stats::median(plot_df$Value, na.rm = TRUE)
  }

  values_interval = c(min(plot_df$Value), round(midpoint_val, digits = 1), max(plot_df$Value))

  # ------------------------------------------------------------------
  # 4. Plot heatmap
  # ------------------------------------------------------------------
  p <- ggplot(plot_df, aes(x = Feature, y = Cluster, fill = Value)) +
    geom_tile(color = "grey70", linewidth = 0.4) +
    ggplot2::scale_fill_gradient2(
      low      = low_color,
      mid      = mid_color,
      high     = high_color,
      midpoint = midpoint_val,
      name     = "",
      breaks=values_interval,
      labels=values_interval,
      limits=c(min(values_interval),max(values_interval))
    ) +
    labs(
      title = "Custom heatmap",
      x = "Features",
      y = "Clusters"
    ) +
    theme_minimal() +
    theme(
      axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1),
      panel.grid = element_blank()
    )

  # Return both the fingerprint matrix and the plot
  return(
    list(
      fingerprint_df = fingerprint_mat,
      plot = p
    )
  )
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
