#' UMAP Embedding and Resolution Parameter Scan
#'
#' Performs dimensionality reduction using UMAP (Uniform Manifold Approximation
#' and Projection) and systematically scans Leiden clustering resolution parameters
#' (gamma) to help identify the optimal resolution for achieving a desired number
#' of clusters. This is the first step in a two-stage clustering workflow.
#'
#' The function creates a UMAP graph from the input data and tests multiple gamma
#' values to show how the number of clusters varies with resolution. This allows
#' users to make an informed decision about which gamma value to use for final
#' clustering.
#'
#' @param data Either a file path (character string) to an Excel/CSV file, or a
#'   data.frame. The last column should contain condition/class labels. If a
#'   data.frame is provided, it will be saved to a temporary CSV file automatically.
#' @param n_neighbors Integer. Number of neighboring points used in UMAP manifold
#'   approximation. Larger values preserve more global structure; smaller values
#'   preserve more local structure. Default: 15.
#' @param min_dist Numeric. Minimum distance between points in UMAP embedding space.
#'   Controls how tightly points can be packed together. Smaller values create
#'   tighter clusters. Range: 0 to 1. Default: 0.1.
#' @param gamma_min Numeric. Minimum resolution (gamma) value to test. Lower gamma
#'   produces fewer, larger clusters. Default: 0.1.
#' @param gamma_max Numeric. Maximum resolution (gamma) value to test. Higher gamma
#'   produces more, smaller clusters. Default: 1.0.
#' @param n_gamma_steps Integer. Number of gamma values to test between gamma_min
#'   and gamma_max. More steps provide finer resolution in the scan. Default: 100.
#' @param save_graph Logical. If TRUE, saves the UMAP graph to disk for use in
#'   subsequent clustering with \code{\link{run_leiden_clustering}}. Default: TRUE.
#'
#' @return A list containing:
#'   \describe{
#'     \item{umap_df}{Data.frame with UMAP coordinates (UMAP1, UMAP2) and original
#'       condition labels (Class) for each sample.}
#'     \item{resolution_scan}{Data.frame showing the relationship between gamma
#'       values and the resulting number of clusters. Use this to select optimal gamma.}
#'     \item{graph_path}{Character string with path to the saved UMAP graph file
#'       (.pkl format). Only included if save_graph=TRUE. Required for subsequent
#'       clustering with \code{\link{run_leiden_clustering}}.}
#'   }
#'
#' @details
#' This function calls a Python script that uses umap-learn and igraph libraries.
#' The UMAP graph is constructed using the specified n_neighbors and min_dist
#' parameters, then Leiden clustering is run with various gamma values to create
#' the resolution scan.
#'
#' After running this function, examine the resolution_scan data frame or plot it
#' using \code{\link{plot_resolution_scan}} to choose an appropriate gamma value
#' for final clustering.
#'
#' @seealso \code{\link{run_leiden_clustering}} for the second step of clustering,
#'   \code{\link{plot_resolution_scan}} for visualizing the scan results,
#'   \code{\link{plot_umap}} for visualizing UMAP embeddings
#'
#' @export
#' @import processx
#' @import reticulate
#' @import tools
#'
#' @examples
#' \dontrun{
#' # Using a file path
#' results <- run_umap_resolution_scan(
#'   data = "path/to/features.csv",
#'   n_neighbors = 15,
#'   gamma_min = 0.1,
#'   gamma_max = 2.0,
#'   n_gamma_steps = 100
#' )
#'
#' # View resolution scan results
#' print(results$resolution_scan)
#' plot_resolution_scan(results$resolution_scan)
#'
#' # Using a data.frame directly
#' results <- run_umap_resolution_scan(data = my_features_df)
#'
#' # Proceed to clustering with selected gamma
#' clusters <- run_leiden_clustering(
#'   umap_data = results$umap_df,
#'   graph_path = results$graph_path,
#'   gamma = 0.42
#' )
#' }
run_umap_resolution_scan <- function(data,
                                     n_neighbors = 15,
                                     min_dist = 0.1,
                                     gamma_min = 0.1,
                                     gamma_max = 1.0,
                                     n_gamma_steps = 100,
                                     save_graph = TRUE) {

  # Handle data input - either path or dataframe
  temp_file_created <- FALSE
  if (is.data.frame(data)) {
    # Create temporary CSV file
    temp_dir <- tempdir()
    data_path <- file.path(temp_dir, paste0("temp_umap_data_", format(Sys.time(), "%Y%m%d_%H%M%S"), ".csv"))
    write.csv(data, data_path, row.names = FALSE)
    temp_file_created <- TRUE
    cat("Data saved to temporary file:", data_path, "\n")
  } else if (is.character(data)) {
    data_path <- normalizePath(path.expand(data), mustWork = TRUE)
    data = read.csv(data_path)
  } else {
    stop("'data' must be either a file path (character) or a dataframe")
  }


  # Locate Python script
  script_path <- system.file("python/umap_resolution_scan.py", package = "RabAnalyser")
  if (!file.exists(script_path)) {
    stop("Python script not found at: ", script_path)
  }

  # Build command arguments
  args <- c(
    script_path,
    data_path,
    as.character(n_neighbors),
    as.character(min_dist),
    as.character(gamma_min),
    as.character(gamma_max),
    as.character(n_gamma_steps)
  )

  cat("Running Python UMAP + Resolution Scan...\n")
  venv_python <- reticulate::virtualenv_python("rabanalyser-venv")
  python_path <- setup_rabanalyser_venv()

  # Use processx::run for synchronous execution with real-time output
  result <- tryCatch({
    processx::run(
      command = venv_python,
      args = args,
      stdout_line_callback = function(line, proc) {
        cat(line, "\n", sep = "")
      },
      stderr_line_callback = function(line, proc) {
        cat("[stderr]", line, "\n", sep = "")
      }
    )
  }, error = function(e) {
    # Clean up temp file if created
    if (temp_file_created && file.exists(data_path)) {
      file.remove(data_path)
    }
    stop("Python execution failed: ", e$message, "\n", call. = FALSE)
  })

  if (result$status != 0) {
    # Clean up temp file if created
    if (temp_file_created && file.exists(data_path)) {
      file.remove(data_path)
    }
    stop("Python script failed with status ", result$status, "\n",
         "stderr: ", result$stderr)
  }

  # Read output CSV files
  output_dir <- dirname(data_path)
  base_name <- tools::file_path_sans_ext(basename(data_path))

  cat("\nReading results...\n")

  results <- list(
    umap_df = read.csv(file.path(output_dir, paste0(base_name, "_umap.csv"))),
    resolution_scan = read.csv(file.path(output_dir, paste0(base_name, "_resolution_scan.csv")))
  )

  results$umap_features = cbind(results$umap_df, data %>% select(-Class) )


  # Store graph path if saved
  if (save_graph) {
    results$graph_path <- file.path(output_dir, paste0(base_name, "_umap_graph.pkl"))
  }

  # Clean up CSV files
  file.remove(file.path(output_dir, paste0(base_name, "_umap.csv")))
  file.remove(file.path(output_dir, paste0(base_name, "_resolution_scan.csv")))

  # Clean up temporary data file if created
  if (temp_file_created && file.exists(data_path)) {
    file.remove(data_path)
  }

  cat("Done!\n")
  if (save_graph) {
    cat("UMAP graph saved to:", results$graph_path, "\n")
  }
  return(results)
}


#' Leiden Clustering with Specified Resolution
#'
#' Performs community detection using the Leiden algorithm on a pre-computed UMAP
#' graph with a user-specified resolution parameter (gamma). This is the second
#' step in a two-stage clustering workflow, following \code{\link{run_umap_resolution_scan}}.
#'
#' The Leiden algorithm is an improved version of the Louvain method for community
#' detection that guarantees well-connected communities. This function uses the
#' UMAP graph created by \code{\link{run_umap_resolution_scan}} and applies Leiden
#' clustering with the specified gamma value. Optionally performs bootstrap stability
#' analysis to assess cluster robustness.
#'
#' @param umap_data Either a file path (character string) to a CSV file containing
#'   UMAP coordinates, or a data.frame with UMAP coordinates (typically from
#'   \code{\link{run_umap_resolution_scan}}). If a data.frame is provided, it will
#'   be saved to a temporary CSV file automatically.
#' @param graph_path Character string. File path to the saved UMAP graph (.pkl format)
#'   created by \code{\link{run_umap_resolution_scan}}. This graph is used for
#'   clustering to ensure consistency with the UMAP embedding.
#' @param gamma Numeric. Leiden resolution parameter controlling the granularity of
#'   clustering. Lower values (e.g., 0.1-0.5) produce fewer, larger clusters; higher
#'   values (e.g., 0.5-2.0) produce more, smaller clusters. Use the resolution scan
#'   results to select an appropriate value. Default: 0.42.
#' @param n_bootstrap Integer. Number of bootstrap iterations for stability analysis.
#'   Each iteration subsamples the data and re-clusters to assess consistency.
#'   More iterations provide more reliable stability estimates but take longer.
#'   Default: 100.
#' @param subsample_prop Numeric. Proportion of samples to include in each bootstrap
#'   iteration (range: 0 to 1). For example, 0.8 means 80% of samples are randomly
#'   selected for each iteration. Default: 0.8.
#' @param stability_analysis Logical. If TRUE, performs bootstrap stability analysis
#'   to quantify how robust each cluster is to random subsampling. Results include
#'   per-cluster stability scores. Default: TRUE.
#'
#' @return A list containing:
#'   \describe{
#'     \item{umap_df}{Data.frame with UMAP coordinates (UMAP1, UMAP2), condition
#'       labels (Class), and cluster assignments (Cluster) for each sample.}
#'     \item{stability}{Data.frame with bootstrap stability scores for each cluster
#'       (only included if stability_analysis=TRUE). Higher scores indicate more
#'       stable/robust clusters. Scores range from 0 to 1.}
#'   }
#'
#' @details
#' This function calls a Python script that uses the leidenalg and igraph libraries.
#' The clustering is performed on the UMAP graph from the first step, ensuring that
#' the cluster structure is consistent with the UMAP visualization.
#'
#' Bootstrap stability analysis works by repeatedly:
#' 1. Randomly sampling a subset of the data (controlled by subsample_prop)
#' 2. Running Leiden clustering on the subset
#' 3. Measuring how consistently samples are assigned to the same clusters
#'
#' Clusters with high stability scores are more reliable and likely represent
#' true biological or technical subpopulations.
#'
#' @seealso \code{\link{run_umap_resolution_scan}} for the first step that creates
#'   the UMAP graph, \code{\link{plot_cluster_stability}} for visualizing stability
#'   results, \code{\link{plot_umap}} for visualizing cluster assignments
#'
#' @export
#' @import processx
#' @import reticulate
#' @import tools
#'
#' @examples
#' \dontrun{
#' # First, run UMAP and resolution scan
#' scan_results <- run_umap_resolution_scan(data = "features.csv")
#'
#' # Examine resolution scan to choose gamma
#' plot_resolution_scan(scan_results$resolution_scan)
#'
#' # Run Leiden clustering with chosen gamma
#' cluster_results <- run_leiden_clustering(
#'   umap_data = scan_results$umap_df,
#'   graph_path = scan_results$graph_path,
#'   gamma = 0.42,
#'   n_bootstrap = 100
#' )
#'
#' # View cluster assignments
#' print(table(cluster_results$umap_df$Cluster))
#'
#' # Plot UMAP colored by clusters
#' plot_umap(cluster_results$umap_df, color_by = "Cluster")
#'
#' # Check cluster stability
#' plot_cluster_stability(cluster_results$stability)
#' }
run_leiden_clustering <- function(umap_data,
                                  graph_path,
                                  gamma = 0.42,
                                  n_bootstrap = 100,
                                  subsample_prop = 0.8,
                                  stability_analysis = TRUE) {

  # Get Python executable from virtualenv
  venv_python <- reticulate::virtualenv_python("rabanalyser-venv")
  python_path <- setup_rabanalyser_venv()

  # Locate Python script
  script_path <- system.file("python/leiden_clustering.py", package = "RabAnalyser")
  if (!file.exists(script_path)) {
    stop("Python script not found at: ", script_path)
  }

  # Temp file creation for dataframes
  umap_file_created <- FALSE
  if (is.data.frame(umap_data)) {
    # Create temporary CSV file
    temp_dir <- tempdir()
    umap_csv_path <- file.path(temp_dir, paste0("umap_data_", format(Sys.time(), "%Y%m%d_%H%M%S"), ".csv"))
    write.csv(umap_data, umap_csv_path, row.names = FALSE)
    umap_file_created <- TRUE
  } else if (is.character(umap_data)) {
    umap_csv_path <- normalizePath(path.expand(umap_data), mustWork = TRUE)
  } else {
    stop("'umap_data' must be either a file path (character) or a dataframe")
  }

  if (is.character(graph_path)) {
    graph_path <- normalizePath(path.expand(graph_path), mustWork = TRUE)
  } else {
    stop("'graph_path' must be a file path (character).")
  }
  ###################
  # Build command arguments
  args <- c(
    script_path,
    umap_csv_path,
    graph_path,
    as.character(gamma),
    as.character(n_bootstrap),
    as.character(subsample_prop)
  )

  cat("Running Python Leiden Clustering...\n")
  # Use processx::run for synchronous execution with real-time output
  result <- tryCatch({
    processx::run(
      command = venv_python,
      args = args,
      stdout_line_callback = function(line, proc) {
        cat(line, "\n", sep = "")
      },
      stderr_line_callback = function(line, proc) {
        cat("[stderr]", line, "\n", sep = "")
      }
    )
  }, error = function(e) {
    stop("Python execution failed: ", e$message, "\n", call. = FALSE)
  })

  if (result$status != 0) {
    stop("Python script failed with status ", result$status, "\n",
         "stderr: ", result$stderr)
  }

  # Read output CSV files
  output_dir <- dirname(umap_csv_path)
  base_name <- tools::file_path_sans_ext(basename(umap_csv_path))
  base_name <- gsub("_umap$", "", base_name)

  cat("\nReading results...\n")

  results <- list(
    umap_df = read.csv(file.path(output_dir, paste0(base_name, "_leiden_clusters.csv")))
  )

  # Clean up CSV file
  file.remove(file.path(output_dir, paste0(base_name, "_leiden_clusters.csv")))

  if (stability_analysis) {
    stability_path <- file.path(output_dir, paste0(base_name, "_stability.csv"))
    if (file.exists(stability_path)) {
      results$stability <- read.csv(stability_path)
      file.remove(stability_path)
    }
  }

  # Clean up temporary data file if created
  if (umap_file_created && file.exists(umap_csv_path)) {
    file.remove(umap_csv_path)
  }

  cat("Done!\n")
  return(results)
}

