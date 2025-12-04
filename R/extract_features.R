#' Extract Rab Spot Features from a Folder (Python Backend via processx)
#'
#' This function performs automated feature extraction from TIFF microscopy images
#' within a specified folder by calling Python as a subprocess. It processes nucleus masks,
#' cell masks, Rab raw signals, and spot masks to extract morphological and spatial
#' features of Rab spots using true multiprocessing with the loky backend.
#'
#' The function expects a folder structure with subdirectories:
#' \describe{
#'   \item{nucleus_mask/}{Folder containing nucleus mask TIFF images}
#'   \item{cell_mask/}{Folder containing cell mask TIFF images}
#'   \item{rab5/}{Folder containing raw Rab signal TIFF images}
#'   \item{rab5_mask/}{Folder containing Rab spot mask TIFF images}
#' }
#'
#' @param root_dir Character. Path to the root directory containing experimental condition folders.
#'                 Each condition folder should contain the subdirectories listed above.
#' @param min_spot_size Numeric. Minimum area (in pixels) for Rab spots to include.
#'                      Spots smaller than this are filtered out. Default is 8.
#' @param neighbor_radius Numeric. Radius (in pixels) for searching neighboring Rab spots.
#'                        Default is 15 pixels (considering typical endosome size ~500-1000 nm).
#' @param conditions Character vector. Specific experimental conditions to process.
#'                   If NULL, all folders in root_dir are processed. Default is NULL.
#' @param python_path Character. Path to a Python executable to use. If NULL, uses the
#'                    package's managed virtualenv. Default is NULL.
#' @param nucleus_folder Character. Name of nucleus mask subfolder. Default "nucleus_mask".
#' @param cell_folder Character. Name of cell mask subfolder. Default "cell_mask".
#' @param rab_folder Character. Name of Rab signal subfolder. Default "rab5".
#' @param spot_folder Character. Name of spot mask subfolder. Default "rab5_mask".
#' @param n_jobs Integer. Number of parallel workers for image processing. If NULL,
#'               uses `parallel::detectCores() - 2` (minimum 1). Set to 1 for
#'               serial processing.
#'
#' @return A list of data frames, one for each condition, containing extracted features:
#'   \describe{
#'     \item{Cell_label}{ID of the cell analyzed}
#'     \item{size}{Area of Rab spot in pixels}
#'     \item{IntegInt}{Sum of pixel intensities within spot}
#'     \item{MeanInt}{Mean pixel intensity within spot}
#'     \item{Circ}{Circularity measure (0-1, where 1 is perfect circle)}
#'     \item{Ecc}{Eccentricity of spot (0-1)}
#'     \item{Elo}{Ratio of major to minor axis}
#'     \item{NucleusDist}{Distance from nucleus centroid (normalized by Feret's diameter)}
#'     \item{NucleusDistOut}{Distance from nucleus outline (normalized by Feret's diameter)}
#'     \item{PMDist}{Distance from cell plasma membrane (normalized by Feret's diameter)}
#'     \item{DNN}{Distance to nearest neighboring spot (normalized by Feret's diameter)}
#'     \item{No}{Number of neighboring spots within radius}
#'   }
#'   Each data frame is also saved as a CSV file in the root_dir.
#'
#' @details
#'
#' This function spawns a Python subprocess using \pkg{processx} that runs the packaged
#' `feature_extraction_wrapper.py` script. A dedicated virtual environment
#' (named `"rabanalyser-venv"` by default) is created on-demand if needed,
#' based on the dependency list in `inst/python/requirements.txt`.
#'
#' The Python backend uses joblib with the loky multiprocessing backend,
#' which provides true parallelism for CPU-bound operations.
#'
#' All distance measures are normalized by Feret's diameter (longest distance across
#' the cell's convex hull) for scale-invariant comparisons.
#'
#' @export
#' @examples
#' \dontrun{
#'   # Process all conditions in a directory
#'   results <- extract_features("/path/to/data/folder")
#'
#'   # Process specific conditions only
#'   results <- extract_features("/path/to/data/folder",
#'                               conditions = c("Control", "Treatment"))
#'
#'   # Use custom parameters
#'   results <- extract_features(
#'     "/path/to/data/folder",
#'     min_spot_size = 10,
#'     neighbor_radius = 20
#'   )
#'
#'   # Use custom folder names
#'   results <- extract_features(
#'     "/path/to/data/folder",
#'     nucleus_folder = "nucleus",
#'     cell_folder = "cell",
#'     rab_folder = "rab_signal",
#'     spot_folder = "rab_spots"
#'   )
#'
#'   # Use parallel processing with 4 workers
#'   results <- extract_features(
#'     "/path/to/data/folder",
#'     n_jobs = 4
#'   )
#' }
#'
#' @export
#' @import processx
#' @import reticulate
extract_features <- function(root_dir,
                            min_spot_size = 8,
                            neighbor_radius = 15,
                            conditions = NULL,
                            python_path = NULL,
                            nucleus_folder = "nucleus_mask",
                            cell_folder = "cell_mask",
                            rab_folder = "rab5",
                            spot_folder = "rab5_mask",
                            n_jobs = NULL) {

  # Call Python backend via system2() for loky multiprocessing support
  return(
    .extract_features_system2(
      root_dir = root_dir,
      min_spot_size = min_spot_size,
      neighbor_radius = neighbor_radius,
      conditions = conditions,
      python_path = python_path,
      nucleus_folder = nucleus_folder,
      cell_folder = cell_folder,
      rab_folder = rab_folder,
      spot_folder = spot_folder,
      n_jobs = n_jobs
    )
  )

}

#' Helper: Call Python feature extraction via processx
#'
#' Executes the packaged Python feature extraction wrapper by spawning
#' a Python subprocess via \pkg{processx}. This provides better process control
#' and real-time stdout/stderr streaming. The virtualenv's Python executable
#' is used directly to ensure all dependencies are available.
#' Results are read from CSV files generated by the Python script.
#'
#' @keywords internal
#' @noRd
.extract_features_system2 <- function(root_dir,
                                      min_spot_size = 8,
                                      neighbor_radius = 15,
                                      conditions = NULL,
                                      python_path = NULL,
                                      nucleus_folder = "nucleus_mask",
                                      cell_folder = "cell_mask",
                                      rab_folder = "rab5",
                                      spot_folder = "rab5_mask",
                                      n_jobs = 1) {

  # Locate packaged Python script
  script_path <- system.file("python/feature_extraction_wrapper.py", package = "RabAnalyser")

  if (!file.exists(script_path)) {
    stop("Python script not found at: ", script_path)
  }

  # Find Python executable from virtualenv
  if (is.null(python_path)) {
    envname <- getOption("RabAnalyser.python.envname", "rabanalyser-venv")
    python_path <- find_rabanalyser_python(envname = envname,update = F)
  }

  if (!file.exists(python_path)) {
    stop("Python executable not found at: ", python_path)
  }

  # Expand ~ and normalize the root_dir path for Python
  root_dir <- normalizePath(path.expand(root_dir), mustWork = FALSE)

  # Determine number of parallel jobs
  if (!is.null(n_jobs)) {
    n_jobs <- min(n_jobs, parallel::detectCores() - 1)
  }

  # Build command-line arguments
  args <- c(
    script_path,
    root_dir,
    as.character(min_spot_size),
    as.character(neighbor_radius),
    as.character(n_jobs),
    nucleus_folder,
    cell_folder,
    rab_folder,
    spot_folder
  )

  # Execute Python script using processx
  cat("Executing Python script with:\n")
  cat("  n_jobs:", n_jobs, "\n")
  cat("  nucleus_folder:", nucleus_folder, "\n")
  cat("  cell_folder:", cell_folder, "\n")
  cat("  rab_folder:", rab_folder, "\n")
  cat("  spot_folder:", spot_folder, "\n\n")

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

  cat("\nPython execution completed with status:", result$status, "\n\n")

  if (result$status != 0) {
    stop("Python script exited with non-zero status: ", result$status)
  }

  # Read results from CSV files generated by Python script
  condition_dirs <- list.dirs(root_dir, recursive = FALSE, full.names = FALSE)

  results <- list()

  for (cond in condition_dirs) {
    csv_path <- file.path(root_dir, paste0(cond, ".csv"))

    if (file.exists(csv_path)) {
      cat("Loading results from:", csv_path, "\n")
      df <- read.csv(csv_path, stringsAsFactors = FALSE)
      results[[cond]] <- df
    } else {
      cat("Note: No CSV file found for condition:", cond, "\n")
    }
  }

  if (length(results) == 0) {
    stop("No results were generated by the Python backend")
  }

  # Filter by requested conditions if specified
  if (!is.null(conditions)) {
    keep <- intersect(names(results), conditions)
    if (!length(keep)) {
      warning("None of the requested conditions were produced by the Python backend.")
      return(list())
    }
    results <- results[keep]
  }

  cat(sprintf("\nSuccessfully loaded features from %d conditions\n", length(results)))

  results
}
