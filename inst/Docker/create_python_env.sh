#!/usr/bin/env bash
set -euo pipefail

# create_python_env.sh
# Create a Python virtual environment for the RabAnalyser package and install
# the required packages for Feature_extraction_V2.
#
# Usage:
#   ./create_python_env.sh [ENV_DIR]
# If ENV_DIR is omitted, a .venv directory will be created in the current working directory.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_DIR="${1:-.venv}"

echo "Creating Python virtual environment at: ${ENV_DIR}"

# Create venv
python3 -m venv "${ENV_DIR}"

# Upgrade pip and install requirements
"${ENV_DIR}/bin/python" -m pip install --upgrade pip
"${ENV_DIR}/bin/python" -m pip install -r "${SCRIPT_DIR}/PythonScripts/requirements.txt"

cat <<'EOF'
Done.
To activate the environment:
  source .venv/bin/activate
To run the feature extraction script:
  .venv/bin/python inst/Docker/PythonScripts/feature_extraction_wrapper.py <root_dir> <min_spot_size> <neighbor_radius>
EOF
