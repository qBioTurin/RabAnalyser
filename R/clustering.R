#' Run UMAP + Leiden Clustering via Python subprocess
#'
#' Performs feature selection, UMAP embedding, Leiden clustering, and stability analysis
#' by calling Python as a subprocess using processx.
#'
#' @param data_path Character. Path to input file (Excel or CSV). Last column should be condition labels.
#' @param feature_threshold Numeric. Correlation threshold for feature filtering. Default 0.7.
#' @param n_neighbors Integer. UMAP n_neighbors parameter. Default 15.
#' @param min_dist Numeric. UMAP min_dist parameter. Default 0.1.
#' @param resolution Numeric. Leiden resolution parameter. Default 0.42.
#' @param n_bootstrap Integer. Number of bootstrap runs for stability. Default 100.
#' @param subsample_prop Numeric. Proportion of samples in each bootstrap. Default 0.8.
#' @param scan_resolutions Logical. If TRUE, scan resolution parameters. Default TRUE.
#' @param stability_analysis Logical. If TRUE, perform stability analysis. Default TRUE.
#' @return List with dataframes:
#'   \describe{
#'     \item{umap_df}{UMAP coordinates with cluster labels and conditions}
#'     \item{resolution_scan}{Resolution vs number of clusters (if scan_resolutions=TRUE)}
#'     \item{stability}{Bootstrap stability results (if stability_analysis=TRUE)}
#'     \item{correlation_before}{Correlation matrix before filtering}
#'     \item{correlation_after}{Correlation matrix after filtering}
#'   }
#' @export
#' @import processx
#'
run_umap_leiden <- function(data_path,
                            n_neighbors = 15,
                            min_dist = 0.1,
                            resolution = 0.42,
                            n_bootstrap = 100,
                            subsample_prop = 0.8,
                            scan_resolutions = TRUE,
                            stability_analysis = TRUE) {


  # Get Python executable from virtualenv
  envname <- getOption("RabAnalyser.python.envname", "rabanalyser-venv")
  python_path <- find_rabanalyser_python(envname = envname, update = F)

  # Locate Python script
  script_path <- system.file("python/umap_leiden_wrapper.py", package = "RabAnalyser")
  if (!file.exists(script_path)) {
    stop("Python script not found at: ", script_path)
  }

  # Expand paths
  data_path <- normalizePath(path.expand(data_path), mustWork = TRUE)

  # Build command arguments
  args <- c(
    script_path,
    data_path,
    as.character(n_neighbors),
    as.character(min_dist),
    as.character(resolution),
    as.character(n_bootstrap),
    as.character(subsample_prop),
    as.character(as.integer(scan_resolutions)),
    as.character(as.integer(stability_analysis))
  )

  cat("Running Python UMAP + Leiden clustering...\n")
  venv_python <- reticulate::virtualenv_python("rabanalyser-venv")

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
  output_dir <- dirname(data_path)
  base_name <- tools::file_path_sans_ext(basename(data_path))

  cat("\nReading results...\n")

  results <- list(
    umap_df = read.csv(file.path(output_dir, paste0(base_name, "_umap_clusters.csv")))
  )
  file.remove(file.path(output_dir, paste0(base_name, "_umap_clusters.csv")))

  if (scan_resolutions) {
    res_scan_path <- file.path(output_dir, paste0(base_name, "_resolution_scan.csv"))
    if (file.exists(res_scan_path)) {
      results$resolution_scan <- read.csv(res_scan_path)
      file.remove(res_scan_path)
    }
  }

  if (stability_analysis) {
    stability_path <- file.path(output_dir, paste0(base_name, "_stability.csv"))
    if (file.exists(stability_path)) {
      results$stability <- read.csv(stability_path)
      file.remove(stability_path)
    }
  }

  cat("Done!\n")
  return(results)
}

