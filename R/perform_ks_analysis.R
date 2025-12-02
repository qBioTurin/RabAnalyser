#' Perform KS test-based analysis between control and comparison matrix
#'
#' @param ctrl_matrix Matrix for control group
#' @param comp_matrix Matrix for comparison group
#' @param features Feature names
#' @param cores Number of cores to use for parallel computing
#'
#' @return A data.frame with KS statistics
#' @export
#' @import parallel

perform_ks_analysis <- function(ctrl_matrix, comp_matrix, features, cores = NULL) {
  if(is.null(cores)) cores = max(parallel::detectCores() - 1, 1)

  ctrl_matrix = ctrl_matrix %>% group_by(ID_image) %>%
    mutate(Cell_label_global = abs(c(1, diff(Cell_label) ) ) )  %>%
    ungroup( ) %>%
    mutate(Cell_label_global = cumsum(Cell_label_global)) %>%
    select(-ID_image,-Cell_label) %>%
    relocate(Cell_label_global)

  comp_matrix = comp_matrix %>% group_by(ID_image) %>%
    mutate(Cell_label_global = abs(c(1, diff(Cell_label) ) ) )  %>%
    ungroup( ) %>%
    mutate(Cell_label_global = cumsum(Cell_label_global)) %>%
    select(-ID_image,-Cell_label) %>%
    relocate(Cell_label_global)

  ctrl <- RabAnalyser::cleaning(ctrl_matrix)
  comp <- RabAnalyser::cleaning(comp_matrix)
  colnames(ctrl) <- names(comp) <- c("Cell_label", features)

  ecdf1List <- lapply(features, function(n) ecdf(ctrl[, n]))
  names(ecdf1List) <- features

  Mcomp <- max(comp[, 1])

  comp_groups <- split(comp, comp[, 1])

  compute_ks <- function(i) {
    sapply(features, function(n) {
      compN <- comp_groups[[as.character(i)]][, n]
      Ref <- ctrl[, n]
      KS <- RabAnalyser::two_sample_signed_ks_statistic(Ref, compN, ecdf1List[[n]])
      KS[2]
    })
  }

  cl <- parallel::makeCluster(cores)
  parallel::clusterExport(cl, c("comp_groups", "ctrl", "ecdf1List", "two_sample_signed_ks_statistic","features"), envir = environment())
  results <- parallel::parLapply(cl, names(comp_groups), compute_ks)
  parallel::stopCluster(cl)

  df <- as.data.frame(do.call(rbind, results))
  names(df) <- features

  return(df)
}
