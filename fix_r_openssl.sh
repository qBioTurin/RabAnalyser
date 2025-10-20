#!/bin/bash
# This script fixes R's OpenSSL 1.1 compatibility on macOS

# Step 1: Set environment variables for shell
export LDFLAGS="-L/usr/local/opt/openssl@1.1/lib"
export CPPFLAGS="-I/usr/local/opt/openssl@1.1/include"
export LD_LIBRARY_PATH="/usr/local/opt/openssl@1.1/lib:$LD_LIBRARY_PATH"
export DYLD_LIBRARY_PATH="/usr/local/opt/openssl@1.1/lib:$DYLD_LIBRARY_PATH"
export PKG_CONFIG_PATH="/usr/local/opt/openssl@1.1/lib/pkgconfig"

# Remove broken packages
rm -rf /usr/local/lib/R/4.5/site-library/openssl
rm -rf /usr/local/lib/R/4.5/site-library/umap

# Install packages that don't depend on openssl first
echo "Installing independent packages..."
R --slave << 'REOF'
packages_basic <- c("dplyr", "tidyr", "ggplot2", "pheatmap", "readxl", "igraph", "randomForest")
install.packages(packages_basic, repos="http://cran.r-project.org")
REOF

echo "Done! Basic packages installed."
echo ""
echo "For other packages that depend on openssl, you may need to:"
echo "1. Use RStudio's Tools > Global Options > Packages > CRAN Mirror"
echo "2. Or install from source with proper flags:"
echo ""
echo "   R CMD install openssl_2.3.4.tar.gz"
echo ""
echo "If you still have issues, consider using a package manager like"
echo "the R-universe binary packages or switching to R 4.4 or earlier."
