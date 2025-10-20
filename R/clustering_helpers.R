#' Generate a violin plot from clustered data
#'
#' @param data Data frame with cluster and feature columns
#' @param feature Feature to visualize
#' @param groupTreat Whether grouping by treatment
#'
#' @return A ggplot2 object
#' @export
#' @import ggplot2
#'
generate_violin_plot <- function(data, feature, groupTreat = FALSE) {

  if (groupTreat) {
    ggplot(data, aes(x = cluster, y = .data[[feature]], color = Treatment, fill = Treatment)) +
      geom_violin(alpha = 0.5) +
      geom_jitter(position = position_jitterdodge()) +
      theme_minimal() +
      ggtitle(paste("Violin Plot of", feature, "by Cluster and Treatment"))
  } else {
    ggplot(data, aes(x = cluster, y = .data[[feature]], color = cluster, fill = cluster)) +
      geom_violin(alpha = 0.5) +
      geom_jitter(width = 0.1) +
      theme_minimal() +
      ggtitle(paste("Violin Plot of", feature, "by Cluster"))
  }
}
