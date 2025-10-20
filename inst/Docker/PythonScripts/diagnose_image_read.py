#!/usr/bin/env python3
"""
Diagnostic script to compare how images are read and identify the issue.
"""

import os
import sys
from pathlib import Path

# Add path for imports
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))

from tifffile import imread
import numpy as np
import MyFunctions_FeatExtract as F

# Load first image
data_dir = Path.home() / "Desktop" / "SimoeTealdiDATA" / "Images"
condition_path = str(data_dir / "Vehicle_Rab5")

print("="*70)
print("COMPARING IMAGE READ METHODS")
print("="*70)

# Load image paths
dsnucleus_mask = F.load_images(condition_path, "nucleus_mask")
print(f"\nFound {len(dsnucleus_mask)} nucleus mask images")
print(f"First image: {dsnucleus_mask[0]}")

# Load with tifffile (Python)
img_tifffile = imread(dsnucleus_mask[0])
print(f"\n--- tifffile (Python) ---")
print(f"Shape: {img_tifffile.shape}")
print(f"Data type: {img_tifffile.dtype}")
print(f"Min value: {np.min(img_tifffile)}")
print(f"Max value: {np.max(img_tifffile)}")
print(f"Mean value: {np.mean(img_tifffile):.2f}")
print(f"Unique values (first 20): {np.unique(img_tifffile)[:20]}")
print(f"Non-zero pixels: {np.count_nonzero(img_tifffile)}")
print(f"Zero pixels: {np.sum(img_tifffile == 0)}")

# Check image properties
print(f"\nFirst 5x5 corner:")
print(img_tifffile[:5, :5])

print(f"\nCenter region (shape {img_tifffile.shape}):")
center_y, center_x = img_tifffile.shape[0]//2, img_tifffile.shape[1]//2
print(img_tifffile[center_y-2:center_y+3, center_x-2:center_x+3])

print("\n" + "="*70)
print("TIFF FILE INFORMATION")
print("="*70)

# Try to read raw TIFF metadata
import tifffile
with tifffile.TiffFile(dsnucleus_mask[0]) as tif:
    print(f"Number of series: {len(tif.series)}")
    print(f"Number of pages: {len(tif.pages)}")
    
    for i, page in enumerate(tif.pages[:3]):  # Show first 3 pages
        print(f"\nPage {i}:")
        print(f"  Shape: {page.shape}")
        print(f"  Data type: {page.dtype}")
        print(f"  Photometric: {page.photometric}")
        print(f"  Bits per sample: {page.bits_per_sample}")
        print(f"  Compression: {page.compression}")

print("\n" + "="*70)
print("RECOMMENDATIONS FOR R (EBImage)")
print("="*70)
print("""
Possible causes of black image in EBImage:
1. Image is stored as inverted (black=data, white=background)
   → Solution: Use !image or 1 - image/max(image)

2. Image data type issue (uint16 not properly converted)
   → Solution: Ensure normalization to [0,1] or [0,255]

3. EBImage interprets photometric differently
   → Solution: Check photometric interpretation (MINISBLACK vs MINISWHITE)

4. Image needs to be rescaled
   → Solution: Use as.numeric() or normalize before display

Check your R code:
- Try: image_normalized <- nucleus_m / max(nucleus_m)
- Try: image_normalized <- !nucleus_m (if inverted)
- Try: print(range(nucleus_m)) to check actual values
""")
