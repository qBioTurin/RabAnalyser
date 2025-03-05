

FilterFeat <- function(df, threshold) {
  # Compute correlation matrix
  correlation_matrix <- cor(df, use = "pairwise.complete.obs")

  # Find feature pairs with correlation above the threshold
  correlated_pairs <- which(abs(correlation_matrix) > threshold, arr.ind = TRUE)
  correlated_pairs <- correlated_pairs[correlated_pairs[, 1] < correlated_pairs[, 2], ]

  removed_features <- c()

  for (i in seq_len(nrow(correlated_pairs))) {
    feature1 <- colnames(df)[correlated_pairs[i, 1]]
    feature2 <- colnames(df)[correlated_pairs[i, 2]]

    if (feature1 %in% removed_features || feature2 %in% removed_features) {
      next
    }

    # Compare variances to decide which feature to remove
    if (var(df[[feature1]], na.rm = TRUE) > var(df[[feature2]], na.rm = TRUE)) {
      removed_features <- c(removed_features, feature2)
    } else {
      removed_features <- c(removed_features, feature1)
    }
  }

  # Return the DataFrame with redundant features removed
  df <- df[, !colnames(df) %in% removed_features]
  return(df)
}

cluster_indexes <-function(data){
  AllIndexes = do.call(rbind,
                       lapply(2:6,function(k){
                         # Perform the kmeans algorithm
                         cl <- kmeans(data, k)
                         df = clusterCrit::intCriteria(as.matrix(data),cl$cluster,"all")
                         data.frame(k = k, as.data.frame(df) )
                       }
                       )
  )


  vals <- vector()
  for(nIndex in names(AllIndexes %>% select(-k))){
    vals = c(vals, clusterCrit::bestCriterion(AllIndexes[[nIndex]],nIndex))
  }

  bestK = sort(table(AllIndexes$k[vals]),decreasing = T)

  return(list(bestK = bestK, AllIndexes = AllIndexes))
}

cluster.generation <- function(data) {

    sil_values <- factoextra::fviz_nbclust(data, kmeans, method = "silhouette")
    allCl = cluster_indexes(data)

  return(list(Data = data, silhouette = sil_values, AllClusteringIndex = allCl))
}
