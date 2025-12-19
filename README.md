# RabAnalyser

**RabAnalyser** is an R package for automated feature extraction, dimensionality reduction, clustering, and statistical analysis of microscopy imaging data. It provides both an interactive Shiny web application and a comprehensive set of R functions for programmatic analysis.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
  - [Prerequisites](#prerequisites)
  - [Installing from GitHub](#installing-from-github)
  - [Python Environment Setup](#python-environment-setup)
- [Quick Start](#quick-start)
  - [Running the Shiny App](#running-the-shiny-app)
  - [Using R Functions](#using-r-functions)
- [Workflows](#workflows)
  - [Feature Extraction](#feature-extraction)
  - [KS Statistical Analysis](#ks-statistical-analysis)
  - [Clustering Analysis](#clustering-analysis)
- [Documentation](#documentation)
- [Citation](#citation)

---

## Features

- **Feature Extraction**: Automated extraction of morphological and intensity features from microscopy images
- **Statistical Analysis**: Kolmogorov-Smirnov (KS) tests for comparing feature distributions between conditions
- **Dimensionality Reduction**: UMAP (Uniform Manifold Approximation and Projection) for visualizing high-dimensional data
- **Clustering**: Leiden algorithm for community detection with resolution parameter optimization
- **Stability Analysis**: Bootstrap-based cluster stability assessment
- **Interactive Visualization**: Built-in Shiny application with modern UI for guided analysis
- **Flexible API**: Comprehensive R functions for custom scripted workflows

---

## Installation

### Prerequisites

**System Requirements:**
- R (≥ 4.0.0)
- Python (3.9 - 3.12, **recommended: 3.11**)
- macOS, Linux, or Windows

**R Packages:**
The package will automatically install required R dependencies, including:
- `reticulate` (for Python integration)
- `shiny`, `bslib`, `shinyFiles`, `shinybusy` (for the web interface)
- `ggplot2`, `dplyr`, `readr`, `readxl`, `DT` (for data manipulation and visualization)

**Python Packages:**
The following Python packages are required and will be installed automatically:
- `numpy`, `pandas`, `scipy`, `matplotlib`
- `scikit-image`, `scikit-learn`, `tifffile`
- `umap-learn`, `python-igraph`, `leidenalg`

### Installing from GitHub

1. **Install the remotes package** (if not already installed):

```r
install.packages("remotes")
```

2. **Install RabAnalyser from GitHub**:

```r
remotes::install_github("qBioTurin/RabAnalyser", ref = "main", dependencies = TRUE)
```

3. **Load the package**:

```r
library(RabAnalyser)
```

### Python Environment Setup

RabAnalyser uses Python for computationally intensive tasks. The package includes an automated setup function:

```r
# Auto-detect compatible Python and create virtual environment
RabAnalyser::setup_rabanalyser_venv()
```

This will:
- Automatically detect a compatible Python installation (3.9-3.12)
- Create a virtual environment named `rabanalyser-venv`
- Install all required Python packages using binary wheels (avoiding compilation issues)

**If you encounter issues**, you can manually specify a Python version:

```r
# Use a specific Python version
RabAnalyser::setup_rabanalyser_venv(
  python = "/usr/local/bin/python3.11",
  update = TRUE
)
```

**To reinstall/update the Python environment:**

```r
RabAnalyser::setup_rabanalyser_venv(update = TRUE)
```

---

## Quick Start

### Running the Shiny App

Launch the interactive web application:

```r
library(RabAnalyser)
RabAnalyser::rabanalyser.run()
```

The app provides a guided workflow through:

  1. **Feature Extraction** from microscopy images
  2. **KS Statistical Analysis** for feature comparison
  3. **Data Clustering** with UMAP and Leiden algorithm
  4. **Visualization** and **Statistical Analysis** of results

### Using R Functions

For scripted analyses, use RabAnalyser functions directly. A complete working example is available in the `demo.R` file included with the package:

```r
# View the demo script location
system.file("demo.R", package = "RabAnalyser")

# Or open it directly
file.edit(system.file("demo.R", package = "RabAnalyser"))
```

**Note:** The demo script uses example microscopy data. To request access to the example dataset used in the demo, please contact the package authors (see [Contact](#contact) section below).

**Basic workflow example:**

```r
library(RabAnalyser)

# 1. Extract features from images
features <- extract_features(
  input_folder = "/path/to/images",
  min_spot_size = 5,
  neighbor_radius = 3
)

# 2. Run UMAP embedding and resolution scan
umap_results <- run_umap_resolution_scan(
  data = features,
  n_neighbors = 15,
  min_dist = 0.1,
  gamma_min = 0.1,
  gamma_max = 2.0
)

# 3. View resolution scan to select optimal gamma
plot_resolution_scan(umap_results$resolution_scan)

# 4. Run Leiden clustering with selected gamma
clusters <- run_leiden_clustering(
  umap_data = umap_results$umap_df,
  graph_path = umap_results$graph_path,
  gamma = 0.42
)

# 5. Visualize results
plot_umap(clusters$umap_df, color_by = "Cluster")
plot_cluster_stability(clusters$stability)
```

---

## Workflows

### Feature Extraction

Extract morphological and intensity features from microscopy images:

```r
features <- extract_features(
  input_folder = "/path/to/experiment/",
  min_spot_size = 5,        # Minimum spot area in pixels
  neighbor_radius = 3,      # Radius for neighbor analysis
  n_jobs = -1,              # Use all CPU cores
  spot_folder = "spots",    # Subfolder with spot masks
  nucleus_folder = "nuclei",
  cell_folder = "cells",
  rab_folder = "rab_channel"
)

# Save features
write.csv(features, "extracted_features.csv", row.names = FALSE)
```

### KS Statistical Analysis

Perform Kolmogorov-Smirnov tests to compare feature distributions:

```r
# Load comparison data
data1 <- read.csv("condition1_features.csv")
data2 <- read.csv("condition2_features.csv")

# Run KS analysis
ks_results <- perform_ks_analysis(
  data1 = data1,
  data2 = data2,
  selected_features = c("feature1", "feature2", "feature3")
)

# Visualize results
plot_statistical_results(ks_results)
```

### Clustering Analysis

Two-step clustering workflow with UMAP and Leiden:

**Step 1: UMAP Embedding and Resolution Scan**

```r
# Load features
features <- read.csv("features.csv")

# Run UMAP and scan resolution parameters
umap_results <- run_umap_resolution_scan(
  data = features,
  n_neighbors = 15,          # UMAP neighbors parameter
  min_dist = 0.1,            # UMAP minimum distance
  gamma_min = 0.1,           # Minimum resolution to test
  gamma_max = 2.0,           # Maximum resolution to test
  n_gamma_steps = 100,       # Number of gamma values to test
  save_graph = TRUE          # Save graph for clustering
)

# Examine resolution scan
print(umap_results$resolution_scan)
plot_resolution_scan(umap_results$resolution_scan)
```

**Step 2: Leiden Clustering with Selected Gamma**

```r
# Run clustering with chosen gamma (e.g., gamma = 0.42)
cluster_results <- run_leiden_clustering(
  umap_data = umap_results$umap_df,
  graph_path = umap_results$graph_path,
  gamma = 0.42,              # Resolution parameter
  n_bootstrap = 100,         # Bootstrap iterations for stability
  subsample_prop = 0.8,      # Subsample proportion
  stability_analysis = TRUE
)

# View cluster assignments
table(cluster_results$umap_df$Cluster)

# Visualize clusters
plot_umap(cluster_results$umap_df, color_by = "Cluster")

# Check stability
plot_cluster_stability(cluster_results$stability)
```

---

## Documentation

Access function documentation in R:

```r
# General help
?RabAnalyser

# Specific functions
?extract_features
?run_umap_resolution_scan
?run_leiden_clustering
?setup_rabanalyser_venv

# List all functions
help(package = "RabAnalyser")
```

**Complete Example Workflow:**

A fully annotated example demonstrating the complete analysis workflow is available in `inst/demo.R`. To view it:

```r
# View demo script location
system.file("demo.R", package = "RabAnalyser")

# Open in editor
file.edit(system.file("demo.R", package = "RabAnalyser"))
```

The demo script includes:
- Feature extraction from microscopy images
- KS statistical analysis between conditions
- UMAP embedding and resolution parameter scanning
- Leiden clustering with stability analysis
- Downstream statistical analysis and visualization

**Note:** To obtain the example dataset used in the demo script, please contact the package authors.

---

## Citation

If you use RabAnalyser in your research, please cite:

```
[Citation information to be added]
```

---

## Contact

For questions, feature requests, bug reports, or to request access to example datasets:

- **GitHub Issues**: https://github.com/qBioTurin/RabAnalyser/issues
- **Repository**: https://github.com/qBioTurin/RabAnalyser
- **Email**: [Contact information to be added]

---

## License

[License information to be added]

---

## Contact

For questions, issues, or contributions, please visit:
https://github.com/qBioTurin/RabAnalyser
