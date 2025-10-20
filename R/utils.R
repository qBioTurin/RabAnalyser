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


#'
#' @param matrix A data frame or matrix.
#'
#' @return A cleaned data frame with complete rows.
#' @export

cleaning <- function(matrix) {
  matrix[complete.cases(matrix), ] %>% as.data.frame()
}

#' Locate Python executable for RabAnalyser
#'
#' Uses the \pkg{reticulate} virtualenv workflow to provision and discover the
#' Python interpreter required by the feature extraction backend. By default an
#' environment named `"rabanalyser-venv"` (configurable via the
#' `RabAnalyser.python.envname` option) will be created on-demand using the
#' dependency list stored in `inst/python/requirements.txt`.
#'
#' @param envname Character scalar giving the virtualenv name. Defaults to the
#'   value of `getOption("RabAnalyser.python.envname", "rabanalyser-venv")`.
#' @param create Logical flag; when `TRUE` (default) the environment is created
#'   and populated if it does not already exist.
#' @param python Optional path to a base Python executable used when creating
#'   the virtualenv.
#'
#' @return Normalized path to the managed Python executable.
#' @export
find_rabanalyser_python <- function(envname = getOption("RabAnalyser.python.envname", "rabanalyser-venv"),
                                    create = TRUE,
                                    python = NULL) {
  if (!requireNamespace("reticulate", quietly = TRUE)) {
    stop("The 'reticulate' package is required. Install it with install.packages('reticulate').")
  }

  if (isTRUE(create)) {
    ensure_rabanalyser_virtualenv(envname = envname, python = python)
  } else if (!reticulate::virtualenv_exists(envname)) {
    stop(
      "Virtual environment '", envname, "' not found. Set create = TRUE or provision it manually."
    )
  }

  # Try to get Python path from reticulate, but add fallback for direct path construction
  python_path <- tryCatch({
    normalizePath(reticulate::virtualenv_python(envname), winslash = "/", mustWork = FALSE)
  }, error = function(e) {
    # Fallback: construct path manually (handles both Unix and Windows)
    venv_root <- reticulate::virtualenv_root()
    # On Unix-like systems: bin/python, on Windows: Scripts/python.exe
    bin_dir <- if (.Platform$OS.type == "windows") "Scripts" else "bin"
    file.path(venv_root, envname, bin_dir, "python")
  })

  # Verify the path exists, try alternatives if not
  if (!file.exists(python_path)) {
    # Determine bin directory for this OS
    bin_dir <- if (.Platform$OS.type == "windows") "Scripts" else "bin"
    exe_names <- if (.Platform$OS.type == "windows") 
                  c("python.exe", "python3.exe") 
                else 
                  c("python", "python3")
    
    # Try alternative paths with different bin directories and executable names
    venv_root <- reticulate::virtualenv_root()
    alt_paths <- c()
    
    # Try with virtualenv root
    for (exe in exe_names) {
      for (bin in c(bin_dir, "bin", "Scripts")) {
        alt_paths <- c(alt_paths, file.path(venv_root, envname, bin, exe))
      }
    }
    
    # Try with .virtualenvs directory (Unix convention)
    home <- Sys.getenv("HOME")
    if (nzchar(home)) {
      for (exe in exe_names) {
        for (bin in c("bin", "Scripts")) {
          alt_paths <- c(alt_paths, file.path(home, ".virtualenvs", envname, bin, exe))
        }
      }
    }
    
    # Try with USERPROFILE (Windows)
    if (.Platform$OS.type == "windows") {
      userprofile <- Sys.getenv("USERPROFILE")
      if (nzchar(userprofile)) {
        for (exe in exe_names) {
          alt_paths <- c(alt_paths, file.path(userprofile, ".virtualenvs", envname, "Scripts", exe))
        }
      }
    }

    for (alt in alt_paths) {
      if (file.exists(alt)) {
        python_path <- alt
        break
      }
    }
  }

  normalizePath(python_path, winslash = "/", mustWork = TRUE)
}

#' @keywords internal
rabanalyser_python_requirements <- function() {
  # Try installed package location first
  req_file <- system.file("python", "requirements.txt", package = "RabAnalyser")

  # Fallback to development location if not found
  if (!nzchar(req_file) || !file.exists(req_file)) {
    req_file <- "inst/python/requirements.txt"
  }

  if (!file.exists(req_file)) {
    stop("requirements.txt not found in package 'RabAnalyser' or development directory")
  }

  lines <- readLines(req_file, warn = FALSE)
  lines <- trimws(lines)
  lines <- lines[lines != ""]
  lines[!startsWith(lines, "#")]
}

#' @keywords internal
ensure_rabanalyser_virtualenv <- function(envname = getOption("RabAnalyser.python.envname", "rabanalyser-venv"),
                                          python = NULL,
                                          packages = rabanalyser_python_requirements()) {
  if (!requireNamespace("reticulate", quietly = TRUE)) {
    stop("The 'reticulate' package is required. Install it with install.packages('reticulate').")
  }

  if (!reticulate::virtualenv_exists(envname)) {
    reticulate::virtualenv_create(envname = envname, python = python)
  }

  packages <- unique(packages[nzchar(packages)])
  if (length(packages) > 0) {
    reticulate::virtualenv_install(envname = envname,
                                   packages = packages,
                                   ignore_installed = FALSE)
  }

  invisible(normalizePath(reticulate::virtualenv_python(envname), winslash = "/", mustWork = TRUE))
}
