"""
Feature Extraction from Rab Spots in Microscopy Images

This script extracts morphological and spatial features from Rab spots in TIFF microscopy images.
It processes nucleus masks, cell masks, Rab raw signals, and spot masks for each experimental condition.

Usage:
    python feature_extraction_wrapper.py <root_dir> <min_spot_size> <neighbor_radius> <output_format>
"""

import os
import sys
import numpy as np
import pandas as pd
from tifffile import imread
from skimage.measure import regionprops
from skimage.morphology import binary_erosion
from scipy.spatial import distance, ConvexHull
import json
from pathlib import Path
from joblib import Parallel, delayed
from multiprocessing import cpu_count

def load_images(condition_path, folder_name):
    """Load all TIFF images from a specific folder."""
    path = os.path.join(condition_path, folder_name)
    if not os.path.exists(path):
        return []
    
    tif_files = sorted([f for f in os.listdir(path) if f.endswith(('.tif', '.tiff'))])
    return [os.path.join(path, f) for f in tif_files]


def synchronize_nucleus_cell(nucleus_m, cell_m):
    """
    Synchronize nucleus and cell masks.
    Reassign nucleus labels to match their corresponding cell labels.
    """
    nucleus_struct = regionprops(nucleus_m)
    nucleus_m_new = np.zeros_like(nucleus_m, dtype=np.int32)
    
    for nucleus_label, prop in enumerate(nucleus_struct, 1):
        centroid = prop.centroid
        x, y = np.round(centroid).astype(int)
        
        # Check bounds
        if 0 <= x < cell_m.shape[0] and 0 <= y < cell_m.shape[1]:
            cell_label = cell_m[x, y]
            if cell_label > 0:
                nucleus_pixels = np.argwhere(nucleus_m == nucleus_label)
                nucleus_m_new[tuple(nucleus_pixels.T)] = cell_label
    
    return nucleus_m_new


def circularity(mask):
    """Calculate circularity of labeled spots."""
    props = regionprops(mask)
    circ_values = []
    
    for p in props:
        perim = p.perimeter
        area = p.area
        
        if perim == 0:
            circ_values.append(np.nan)
        else:
            r = perim / (2 * np.pi) + 0.5
            correction = (1 - 0.5 / r) ** 2
            circ = (4 * np.pi * area / (perim ** 2)) * correction
            circ_values.append(min(circ, 1.0))  # Cap at 1
    
    return np.array(circ_values)


def feret_diameter(cell_mask):
    """Calculate Feret's diameter of a cell mask."""
    contours = np.argwhere(cell_mask > 0)
    if len(contours) < 2:
        return 1.0
    
    distances = distance.pdist(contours)
    return np.max(distances) if len(distances) > 0 else 1.0


def distance_from_nucleus_centroid(nuc_centroid, cell_mask, spot_mask, feret_d):
    """Distance from nucleus centroid normalized by Feret's diameter."""
    spot_props = regionprops(spot_mask)
    distances = []
    
    for prop in spot_props:
        centroid = prop.centroid
        if not (np.isnan(centroid[0]) or np.isnan(centroid[1])):
            d = np.sqrt((centroid[0] - nuc_centroid[0])**2 + 
                       (centroid[1] - nuc_centroid[1])**2) / feret_d
            distances.append(d)
    
    return np.array(distances)


def distance_from_nucleus_outline(nucleus_mask, cell_mask, spot_mask, feret_d):
    """Distance from nucleus outline normalized by Feret's diameter."""
    nuc_outline = nucleus_mask ^ binary_erosion(nucleus_mask)
    outline_coords = np.column_stack(np.nonzero(nuc_outline))
    
    spot_props = regionprops(spot_mask)
    distances = []
    
    for prop in spot_props:
        centroid = prop.centroid
        if len(outline_coords) > 0:
            dists = np.sqrt((outline_coords[:, 0] - centroid[0])**2 + 
                           (outline_coords[:, 1] - centroid[1])**2)
            distances.append(np.min(dists) / feret_d)
        else:
            distances.append(np.nan)
    
    return np.array(distances)


def distance_from_plasma_membrane(cell_mask, spot_mask, feret_d):
    """Distance from plasma membrane normalized by Feret's diameter."""
    bw_cell = (cell_mask > 0).astype(np.uint8)
    cell_perim = bw_cell ^ binary_erosion(bw_cell)
    perim_coords = np.column_stack(np.nonzero(cell_perim))
    
    spot_props = regionprops(spot_mask)
    distances = []
    
    for prop in spot_props:
        centroid = prop.centroid
        if len(perim_coords) > 0:
            dists = np.sqrt((perim_coords[:, 0] - centroid[0])**2 + 
                           (perim_coords[:, 1] - centroid[1])**2)
            distances.append(np.min(dists) / feret_d)
        else:
            distances.append(np.nan)
    
    return np.array(distances)


def nearest_neighbor_distance(cell_mask, spot_mask, feret_d):
    """Distance to nearest neighboring spot normalized by Feret's diameter."""
    spot_props = regionprops(spot_mask)
    distances = []
    
    for idx, prop in enumerate(spot_props):
        centroid = prop.centroid
        other_spots = [sp for i, sp in enumerate(spot_props) if i != idx]
        
        if other_spots:
            other_dists = [np.sqrt((sp.centroid[0] - centroid[0])**2 + 
                                   (sp.centroid[1] - centroid[1])**2) 
                          for sp in other_spots]
            distances.append(np.min(other_dists) / feret_d)
        else:
            distances.append(np.nan)
    
    return np.array(distances)


def number_of_neighbors(spot_mask, neighbor_radius):
    """Count neighboring spots within specified radius."""
    spot_props = regionprops(spot_mask)
    neighbor_counts = []
    
    for idx, prop in enumerate(spot_props):
        centroid = prop.centroid
        other_spots = [sp for i, sp in enumerate(spot_props) if i != idx]
        
        count = 0
        for sp in other_spots:
            dist = np.sqrt((sp.centroid[0] - centroid[0])**2 + 
                          (sp.centroid[1] - centroid[1])**2)
            if dist <= neighbor_radius:
                count += 1
        
        neighbor_counts.append(count)
    
    return np.array(neighbor_counts)


def extract_features_from_image(nucleus_m, cell_m, rab, spot, min_spot_size=8, neighbor_radius=15):
    """Extract all features from a single image."""
    
    # Synchronize nucleus and cell masks
    nucleus_m_new = synchronize_nucleus_cell(nucleus_m, cell_m)
    
    all_features = []
    max_cell = int(np.max(cell_m))
    
    for cell_label in range(1, max_cell + 1):
        mask_cell = (cell_m == cell_label)
        mask_nuc = (nucleus_m_new == cell_label)
        
        if np.sum(mask_cell) == 0:
            continue
        
        # Extract spots for this cell
        spotnew = spot.copy()
        spotnew[~mask_cell] = 0
        
        cell_m_new = cell_m.copy()
        cell_m_new[~mask_cell] = 0
        
        # Filter small spots
        spot_props = regionprops(spotnew)
        small_labels = [prop.label for prop in spot_props if prop.area < min_spot_size]
        
        spotf = spotnew.copy()
        for lbl in small_labels:
            spotf[spotf == lbl] = 0
        
        # Skip if no valid spots
        if len(np.unique(spotf)) - 1 <= 0:
            continue
        
        # Get Feret's diameter
        feret_d = feret_diameter(cell_m_new)
        if feret_d == 0:
            feret_d = 1.0
        
        # Extract features
        area_props = regionprops(spotf)
        area = np.array([prop.area for prop in area_props])
        
        # Integral intensity
        intsum_props = regionprops(spotf, intensity_image=rab)
        intsum = np.array([np.sum(prop.intensity_image) for prop in intsum_props])
        
        # Mean intensity
        mean_int_props = regionprops(spotf, intensity_image=rab)
        mean_int = np.array([prop.mean_intensity for prop in mean_int_props])
        
        # Circularity
        circ = circularity(spotf)
        
        # Eccentricity
        ecc_props = regionprops(spotf)
        ecc = np.array([prop.eccentricity for prop in ecc_props])
        
        # Elongation
        elo = []
        for p in ecc_props:
            if p.minor_axis_length > 0:
                elong = p.major_axis_length / p.minor_axis_length
            else:
                elong = np.nan
            elo.append(elong)
        elo = np.array(elo)
        
        # Distance from nucleus centroid
        nuc_props = regionprops(nucleus_m_new)
        if cell_label - 1 < len(nuc_props):
            nuc_centroid = nuc_props[cell_label - 1].centroid
            dist_nuc = distance_from_nucleus_centroid(nuc_centroid, cell_m_new, spotf, feret_d)
        else:
            dist_nuc = np.full(len(area), np.nan)
        
        # Distance from nucleus outline
        dist_nuc_out = distance_from_nucleus_outline(mask_nuc, cell_m_new, spotf, feret_d)
        
        # Distance from plasma membrane
        dist_pm = distance_from_plasma_membrane(cell_m_new, spotf, feret_d)
        
        # Distance to nearest spot
        dist_nn = nearest_neighbor_distance(cell_m_new, spotf, feret_d)
        
        # Number of neighbors
        num_neigh = number_of_neighbors(spotf, neighbor_radius)
        
        # Combine all features
        cell_features = np.column_stack([
            np.full(len(area), cell_label),  # cell_id
            area,
            intsum,
            mean_int,
            circ,
            ecc,
            elo,
            dist_nuc,
            dist_nuc_out,
            dist_pm,
            dist_nn,
            num_neigh
        ])
        
        all_features.append(cell_features)
    
    if len(all_features) == 0:
        return np.empty((0, 12))
    
    return np.vstack(all_features)


def _process_image_worker(nucleus_file, cell_file, rab_file, spot_file, min_spot_size, neighbor_radius, image_idx, n_images):
    """Worker function to process a single image. Module-level for proper serialization."""
    print(f"  Processing image {image_idx+1}/{n_images}")
    try:
        nucleus_m = imread(nucleus_file)
        cell_m = imread(cell_file)
        rab = imread(rab_file)
        spot = imread(spot_file)
        
        img_features = extract_features_from_image(
            nucleus_m, cell_m, rab, spot,
            min_spot_size=min_spot_size,
            neighbor_radius=neighbor_radius
        )
        
        return (image_idx, img_features if img_features.shape[0] > 0 else None)
    
    except Exception as e:
        print(f"  Error processing image {image_idx+1}: {str(e)}")
        return (image_idx, None)


def _process_single_image(args):
    """Helper function to process a single image (for multiprocessing)."""
    i, nucleus_file, cell_file, rab_file, spot_file, min_spot_size, neighbor_radius = args
    
    try:
        nucleus_m = imread(nucleus_file)
        cell_m = imread(cell_file)
        rab = imread(rab_file)
        spot = imread(spot_file)
        
        img_features = extract_features_from_image(
            nucleus_m, cell_m, rab, spot,
            min_spot_size=min_spot_size,
            neighbor_radius=neighbor_radius
        )
        
        return (i, img_features if img_features.shape[0] > 0 else None)
    
    except Exception as e:
        print(f"  Error processing image {i+1}: {str(e)}")
        return (i, None)



def process_condition(condition_path, min_spot_size=8, neighbor_radius=15, n_jobs=None,
                      nucleus_folder="nucleus_mask", cell_folder="cell_mask",
                      rab_folder="rab5", spot_folder="rab5_mask"):
    """Process all images in a condition folder.
    
    Args:
        condition_path: Path to condition folder
        min_spot_size: Minimum spot area threshold
        neighbor_radius: Radius for neighbor detection
        n_jobs: Number of parallel jobs. If None, uses cpu_count() - 2 (minimum 1). Set to 1 for serial processing.
        nucleus_folder: Name of the folder containing nucleus masks (default: "nucleus_mask")
        cell_folder: Name of the folder containing cell masks (default: "cell_mask")
        rab_folder: Name of the folder containing Rab signals (default: "rab5")
        spot_folder: Name of the folder containing spot masks (default: "rab5_mask")
    """
    
    # Load image paths using custom folder names
    nucleus_files = load_images(condition_path, nucleus_folder)
    cell_files = load_images(condition_path, cell_folder)
    rab_files = load_images(condition_path, rab_folder)
    spot_files = load_images(condition_path, spot_folder)
    
    n_images = min(len(nucleus_files), len(cell_files), len(rab_files), len(spot_files))

    if n_images == 0:
        return np.empty((0, 12))
    
    # Use joblib for cross-platform parallel processing
    # Use 'loky' backend for true multiprocessing (when called directly from Python)
    results = Parallel(n_jobs=n_jobs, backend='loky')(
        delayed(_process_image_worker)(
            nucleus_files[i], cell_files[i], rab_files[i], spot_files[i],
            min_spot_size, neighbor_radius, i, n_images
        )
        for i in range(n_images)
    )
    
    # Collect non-empty results
    all_condition_features = []
    for i, img_features in results:
        if img_features is not None:
            all_condition_features.append(img_features)
    
    if len(all_condition_features) == 0:
        return np.empty((0, 12))
    
    return np.vstack(all_condition_features)


def main(root_dir, min_spot_size=8, neighbor_radius=15, n_jobs=None, output_format="csv",
          nucleus_folder="nucleus_mask", cell_folder="cell_mask",
          rab_folder="rab5", spot_folder="rab5_mask"):
    """Main function to process all conditions.
    
    Args:
        root_dir: Root directory containing condition folders
        min_spot_size: Minimum spot area threshold
        neighbor_radius: Radius for neighbor detection
        n_jobs: Number of parallel jobs. If None, uses cpu_count() - 2 (minimum 1)
        output_format: Output format (currently only 'csv' supported)
        nucleus_folder: Name of the folder containing nucleus masks (default: "nucleus_mask")
        cell_folder: Name of the folder containing cell masks (default: "cell_mask")
        rab_folder: Name of the folder containing Rab signals (default: "rab5")
        spot_folder: Name of the folder containing spot masks (default: "rab5_mask")
    """
    
    print(f"Starting feature extraction in: {root_dir}")
    
    if not os.path.exists(root_dir):
        print(f"Error: root_dir does not exist: {root_dir}")
        sys.exit(1)
    
    # Get condition folders
    conditions = [d for d in os.listdir(root_dir) 
                  if os.path.isdir(os.path.join(root_dir, d)) and not d.startswith('.')]
    
    if len(conditions) == 0:
        print("Error: No condition folders found")
        sys.exit(1)
    
    # Determine number of parallel jobs
    if n_jobs is None:
        n_jobs = max(1, cpu_count() - 2)
    
    print(f"Using {n_jobs} parallel workers (detected {cpu_count()} CPUs)")
    
    column_names = ["Cell_label", "size", "IntegInt", "MeanInt", "Circ", "Ecc", "Elo",
                    "NucleusDist", "NucleusDistOut", "PMDist", "DNN", "No"]
    
    results = {}
    
    for condition in conditions:
        print(f"\nProcessing condition: {condition}")
        condition_path = os.path.join(root_dir, condition)
        
        features = process_condition(condition_path, min_spot_size, neighbor_radius, n_jobs,
                                    nucleus_folder, cell_folder, rab_folder, spot_folder)
        
        if features.shape[0] == 0:
            print(f"  No features extracted for {condition}")
            continue
        
        df = pd.DataFrame(features, columns=column_names)
        results[condition] = df
        
        # Save to CSV or Excel
        output_file = os.path.join(root_dir, f"{condition}.csv")
        df.to_csv(output_file, index=False)
        print(f"  Saved to: {output_file}")
    
    print(f"\nProcessing complete. {len(results)} conditions processed.")
    
    # Return results as JSON for R to parse
    json_results = {
        condition: df.to_dict('records') 
        for condition, df in results.items()
    }
    
    return json_results


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python feature_extraction_wrapper.py <root_dir> [min_spot_size] [neighbor_radius] [n_jobs] [nucleus_folder] [cell_folder] [rab_folder] [spot_folder]")
        sys.exit(1)
    
    root_dir = sys.argv[1]
    min_spot_size = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    neighbor_radius = int(sys.argv[3]) if len(sys.argv) > 3 else 15
    n_jobs = int(sys.argv[4]) if len(sys.argv) > 4 else None
    nucleus_folder = sys.argv[5] if len(sys.argv) > 5 else "nucleus_mask"
    cell_folder = sys.argv[6] if len(sys.argv) > 6 else "cell_mask"
    rab_folder = sys.argv[7] if len(sys.argv) > 7 else "rab5"
    spot_folder = sys.argv[8] if len(sys.argv) > 8 else "rab5_mask"
    
    results = main(root_dir, min_spot_size, neighbor_radius, n_jobs, "csv",
                   nucleus_folder, cell_folder, rab_folder, spot_folder)
