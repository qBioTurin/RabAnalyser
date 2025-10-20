#' Filter Highly Correlated Features
#'
#' Removes redundant features from a data frame by eliminating one feature in each highly correlated pair.
#'
#' @param df A data frame with numeric features.
#' @param threshold Correlation threshold above which features are considered redundant.
#'
#' @return A data frame with reduced features.
#' @export

filter_features <- function(df, threshold) {
  correlation_matrix <- cor(df, use = "pairwise.complete.obs")
  correlated_pairs <- which(abs(correlation_matrix) > threshold, arr.ind = TRUE)
  correlated_pairs <- correlated_pairs[correlated_pairs[, 1] < correlated_pairs[, 2], ]

  removed_features <- c()
  for (i in seq_len(nrow(correlated_pairs))) {
    f1 <- colnames(df)[correlated_pairs[i, 1]]
    f2 <- colnames(df)[correlated_pairs[i, 2]]
    if (f1 %in% removed_features || f2 %in% removed_features) next
    # Compare variances to decide which feature to remove
    if (var(df[[f1]], na.rm = TRUE) > var(df[[f2]], na.rm = TRUE)) {
      removed_features <- c(removed_features, f2)
    } else {
      removed_features <- c(removed_features, f1)
    }
  }
  # Return the DataFrame with redundant features removed
  return(df[, !colnames(df) %in% removed_features])
}
