#' Setup RabAnalyser Python Virtual Environment
#'
#' Creates or updates a Python virtual environment with all packages required
#' for RabAnalyser's feature extraction and clustering workflows. Automatically
#' detects a compatible Python installation (versions 3.9-3.12, with preference
#' for 3.11) and installs packages using pre-compiled binary wheels to avoid
#' compilation errors (particularly with llvmlite/numba dependencies).
#'
#' This function handles the entire Python environment setup process, making it
#' easy to get RabAnalyser's Python backend working correctly. It will:
#' \itemize{
#'   \item Auto-detect a compatible Python installation on your system
#'   \item Create a virtual environment in the standard reticulate location
#'   \item Install all required packages (numpy, pandas, scipy, scikit-image,
#'     scikit-learn, umap-learn, python-igraph, leidenalg, etc.)
#'   \item Use binary wheels to avoid compilation issues
#' }
#'
#' @param envname Character. Name of the virtual environment. The environment
#'   will be created in the standard reticulate virtualenv directory (typically
#'   ~/.virtualenvs/ on Unix systems). Default: "rabanalyser-venv".
#' @param python Character. Full path to a Python executable to use for creating
#'   the virtual environment. If NULL (default), the function will automatically
#'   search for a compatible Python installation (3.9-3.12) on your system.
#'   Recommended: let auto-detection find Python 3.11 for best compatibility.
#' @param update Logical. If TRUE, removes any existing virtual environment with
#'   the same name and creates a fresh one. Use this to fix a broken environment
#'   or upgrade packages. Default: FALSE.
#' @param create Logical. If TRUE, creates the virtual environment if it doesn't
#'   already exist. Set to FALSE to check for existence without creating.
#'   Default: TRUE.
#'
#' @return Character string containing the normalized path to the Python executable
#'   in the virtual environment. This path can be passed to other functions that
#'   need to run Python code.
#'
#' @details
#' The function installs packages in groups to minimize dependency conflicts:
#' \enumerate{
#'   \item Core scientific packages: numpy, pandas, scipy, matplotlib, joblib, scikit-learn
#'   \item Imaging packages: scikit-image, tifffile
#'   \item Graph packages: python-igraph, leidenalg
#'   \item UMAP and dependencies: umap-learn (includes numba and llvmlite)
#' }
#'
#' All packages are installed with the --prefer-binary flag to use pre-compiled
#' wheels instead of building from source, which avoids common compilation errors
#' with LLVM-dependent packages like llvmlite.
#'
#' If the virtual environment already exists and create=TRUE, the function will
#' verify it exists and return the path without reinstalling packages. Use
#' update=TRUE to force recreation.
#'
#' @section Python Version Compatibility:
#' RabAnalyser works best with Python 3.11, but supports 3.9-3.12. The auto-detection
#' prioritizes versions in this order: 3.11, 3.10, 3.12, 3.9. Python 3.13+ is not
#' recommended as some packages may not have binary wheels available yet.
#'
#' @section Troubleshooting:
#' If you encounter issues:
#' \itemize{
#'   \item Try \code{setup_rabanalyser_venv(update = TRUE)} to recreate the environment
#'   \item Ensure you have Python 3.9-3.12 installed (check with \code{python3 --version})
#'   \item On macOS, install Python via Homebrew: \code{brew install python@3.11}
#'   \item Manually specify Python: \code{setup_rabanalyser_venv(python = "/path/to/python3.11")}
#' }
#'
#' @seealso \code{\link{find_rabanalyser_python}} for the legacy interface,
#'   \code{\link[reticulate]{virtualenv_create}} for the underlying reticulate function
#'
#' @export
#' @import reticulate
#'
#' @examples
#' \dontrun{
#' # Auto-detect Python and create virtual environment
#' python_path <- setup_rabanalyser_venv()
#'
#' # Force recreation with auto-detected Python
#' python_path <- setup_rabanalyser_venv(update = TRUE)
#'
#' # Use a specific Python version
#' python_path <- setup_rabanalyser_venv(
#'   python = "/usr/local/bin/python3.11",
#'   update = TRUE
#' )
#'
#' # Check if environment exists without creating it
#' python_path <- setup_rabanalyser_venv(create = FALSE)
#' }
setup_rabanalyser_venv <- function(envname = "rabanalyser-venv",
                                   python = NULL,
                                   update = FALSE,
                                   create = TRUE) {

  cat("=== RabAnalyser Virtual Environment Setup ===\n")

  # Auto-detect compatible Python if not specified
  if (is.null(python)) {
    cat("Detecting compatible Python installation...\n")
    python <- detect_compatible_python()
    if (is.null(python)) {
      stop("No compatible Python installation found. Please install Python 3.9-3.12 or specify the path manually.")
    }
  } else {
    cat("Using specified Python:", python, "\n")
  }

  # If update flag is set, remove existing virtualenv
  if (update) {
    cat("Update flag set: removing existing virtualenv...\n")
    tryCatch({
      reticulate::virtualenv_remove(envname, confirm = FALSE)
      cat("Virtualenv removed successfully.\n")
    }, error = function(e) {
      cat("No existing virtualenv to remove or removal failed.\n")
    })
  }

  # Check if virtualenv exists
  venv_exists <- tryCatch({
    reticulate::virtualenv_exists(envname)
  }, error = function(e) {
    FALSE
  })

  # Create virtualenv if needed
  if (!venv_exists) {
    if (!create) {
      stop("Virtual environment does not exist and create=FALSE. Set create=TRUE to create it.")
    }

    cat("Creating virtual environment:", envname, "\n")
    cat("Using Python:", python, "\n")
    reticulate::virtualenv_create(envname = envname, python = python)

    # Install packages
    cat("\n=== Installing Python Packages ===\n")
    venv_python <- reticulate::virtualenv_python(envname)

    # Upgrade pip, setuptools, and wheel
    cat("Upgrading pip, setuptools, and wheel...\n")
    system2(venv_python, c("-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"))

    # Install packages in groups to avoid conflicts
    # Group 1: Core scientific packages
    cat("\n[1/4] Installing core scientific packages...\n")
    result1 <- system2(venv_python, c("-m", "pip", "install", "--prefer-binary",
                                      "numpy", "pandas", "scipy", "matplotlib", "joblib", "scikit-learn"))

    # Group 2: Imaging packages
    cat("\n[2/4] Installing imaging packages...\n")
    result2 <- system2(venv_python, c("-m", "pip", "install", "--prefer-binary",
                                      "scikit-image", "tifffile"))

    # Group 3: Graph packages
    cat("\n[3/4] Installing graph analysis packages...\n")
    result3 <- system2(venv_python, c("-m", "pip", "install", "--prefer-binary",
                                      "python-igraph", "leidenalg"))

    # Group 4: UMAP (which requires numba/llvmlite)
    cat("\n[4/4] Installing UMAP...\n")
    result4 <- system2(venv_python, c("-m", "pip", "install", "--prefer-binary", "umap-learn"))

    # Check for any installation failures
    if (any(c(result1, result2, result3, result4) != 0)) {
      warning("Some packages may have failed to install. Check the output above.")
    } else {
      cat("\n✓ Package installation complete!\n")
    }
  } else {
    cat("Virtual environment already exists:", envname, "\n")
  }

  # Get and verify Python path
  python_path <- tryCatch({
    normalizePath(reticulate::virtualenv_python(envname), winslash = "/", mustWork = FALSE)
  }, error = function(e) {
    # Fallback: construct path manually
    venv_root <- reticulate::virtualenv_root()
    bin_dir <- if (.Platform$OS.type == "windows") "Scripts" else "bin"
    file.path(venv_root, envname, bin_dir, "python")
  })

  if (!file.exists(python_path)) {
    stop("Virtual environment created but Python executable not found at: ", python_path)
  }

  cat("\n=== Setup Complete ===\n")
  cat("Python path:", python_path, "\n")

  invisible(normalizePath(python_path, winslash = "/", mustWork = TRUE))
}

#' Remove Rows with Missing Values
#'
#' Cleans #' @keywords internal
rabanalyser_python_requirements <- function() {
  req_file <- system.file("python", "requirements.txt", package = "RabAnalyser")

  if (!nzchar(req_file) || !file.exists(req_file)) {
    stop("requirements.txt not found in package 'RabAnalyser'")
  }

  lines <- readLines(req_file, warn = FALSE)
  lines <- trimws(lines)
  lines <- lines[lines != ""]
  lines[!startsWith(lines, "#")]
}

#' Cleaning function to remove rows with missing values
#'
#' @param matrix A data frame or matrix.
#'
#' @return A cleaned data frame with complete rows.
#' @export

cleaning <- function(matrix) {
  matrix[complete.cases(matrix), ] %>% as.data.frame()
}

#' Detect Compatible Python Installation
#'
#' Automatically detects a suitable Python installation (3.9-3.12) for creating
#' the RabAnalyser virtual environment. Prioritizes Python 3.11 for best compatibility.
#'
#' @return Path to a compatible Python executable, or NULL if none found.
#' @keywords internal
detect_compatible_python <- function() {
  # Preferred Python versions in order (3.11 is most stable for our packages)
  preferred_versions <- c("3.11", "3.10", "3.12", "3.9")

  # Common Python executable names
  py_names <- c("python3.11", "python3.10", "python3.12", "python3.9", "python3", "python")

  # Common installation paths
  search_paths <- c(
    "/usr/local/bin",
    "/usr/bin",
    "/opt/homebrew/bin",
    "/Library/Frameworks/Python.framework/Versions/*/bin",
    file.path(Sys.getenv("HOME"), ".pyenv/shims"),
    Sys.getenv("PATH")
  )

  if (.Platform$OS.type == "windows") {
    search_paths <- c(
      file.path(Sys.getenv("LOCALAPPDATA"), "Programs/Python/Python3*/"),
      file.path(Sys.getenv("PROGRAMFILES"), "Python3*/"),
      "C:/Python3*/"
    )
    py_names <- paste0(py_names, ".exe")
  }

  # Function to get Python version
  get_python_version <- function(py_path) {
    tryCatch({
      version_output <- system2(py_path, c("--version"), stdout = TRUE, stderr = TRUE)
      # Extract version number (e.g., "Python 3.11.5" -> "3.11")
      if (grepl("Python [0-9]", version_output[1])) {
        version <- sub(".*Python ([0-9]+\\.[0-9]+).*", "\\1", version_output[1])
        return(version)
      }
      return(NULL)
    }, error = function(e) NULL)
  }

  # Try to find Python using 'which' or 'where' command
  for (version in preferred_versions) {
    py_name <- paste0("python", version)
    if (.Platform$OS.type == "windows") {
      py_path <- tryCatch(system2("where", py_name, stdout = TRUE, stderr = FALSE)[1],
                         error = function(e) NULL)
    } else {
      py_path <- tryCatch(system2("which", py_name, stdout = TRUE, stderr = FALSE),
                         error = function(e) NULL)
    }

    if (!is.null(py_path) && length(py_path) > 0 && file.exists(py_path)) {
      cat("Found compatible Python:", py_path, "\n")
      return(py_path)
    }
  }

  # Manual search in common paths
  for (path in search_paths) {
    # Expand wildcards in path
    expanded_paths <- Sys.glob(path)
    for (search_dir in expanded_paths) {
      if (!dir.exists(search_dir)) next

      for (py_name in py_names) {
        py_path <- file.path(search_dir, py_name)
        if (file.exists(py_path)) {
          version <- get_python_version(py_path)
          if (!is.null(version) && version %in% preferred_versions) {
            cat("Found compatible Python:", py_path, "(version", version, ")\n")
            return(py_path)
          }
        }
      }
    }
  }

  cat("Warning: No compatible Python installation (3.9-3.12) found.\n")
  cat("Please install Python 3.11 for best compatibility.\n")
  return(NULL)
}
