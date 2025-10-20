# Python environment for RabAnalyser — Feature Extraction V2

This directory contains a reproducible Python environment configuration and helper script
for running the Feature_extraction_V2 Python implementation from the RabAnalyser package.

Files:
- `PythonScripts/requirements.txt` — pinned Python dependencies required by the feature-extraction scripts.
- `create_python_env.sh` — convenience script to create a virtual environment and install the required packages.

Quick setup (from package root):

1. Create the virtual environment (default: `.venv` in package root):

```bash
bash inst/Docker/create_python_env.sh .venv
```

2. Activate the environment:

```bash
source .venv/bin/activate
```

3. Run the feature extraction script (example):

```bash
python inst/Docker/PythonScripts/feature_extraction_wrapper.py \
  /path/to/data/root 8 15
```

Notes for integration from R
----------------------------
If you want to call the Python script from R (for example from `extract_features()`), use the Python executable
inside the environment you created. Example R snippet:

```r
# Locate python executable created by the helper script (default .venv)
venv_dir <- file.path(getwd(), ".venv")
python_exec <- if (.Platform$OS.type == "windows") {
  file.path(venv_dir, "Scripts", "python.exe")
} else {
  file.path(venv_dir, "bin", "python")
}

# Run the script via system2
script <- system.file("Docker/PythonScripts/feature_extraction_wrapper.py", package = "RabAnalyser")
root_dir <- "/path/to/data/root"
cmd_args <- c(script, root_dir, 8, 15)
res <- system2(python_exec, args = cmd_args, stdout = TRUE, stderr = TRUE)
cat(res, sep = "\n")
```

If you package and distribute RabAnalyser, you should document the Python environment setup in your package README or vignettes.
