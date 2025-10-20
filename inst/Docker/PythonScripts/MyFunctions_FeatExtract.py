import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import mode
import os
from glob import glob
from skimage import measure
from scipy.spatial import ConvexHull, distance
from skimage.morphology import binary_erosion
from skimage.measure import regionprops

def load_images(condition_path,folder_name):
  
    """

    Parameters
    ----------
    condition_path : path where files are stored
        
    folder_name : name of the forlder where input files are stored

    Returns
    -------
    
    single images to be used as inputs
    

    """
    
    
    path = os.path.join(condition_path, folder_name)
    tif_files = glob(os.path.join(path, "*.tif"))
    tiff_files = glob(os.path.join(path, "*.tiff"))
    return sorted(tif_files + tiff_files)



def circularity(mask):
    
    
    """
    Compute the circularity of labeled Rab spot. This is a corrected version 
    to avoid values that are higher than 1. They are meaningless.
    
    Formula:
    circularity = (4π * area / perimeter²) * (1 - 0.5 / r)²
    where r = perimeter / (2π) + 0.5
    
    Inputs:
    - mask: labeled image of Rab spots
  
    Returns:
    - result: 1D NumPy array of the circularities of Rab spots
    """
   
    props = regionprops(mask)
    circ_values = []

    for p in props:
        perim = p.perimeter
        area = p.area

        if perim == 0:
            circ_values.append(np.nan)
            continue

        r = perim / (2 * np.pi) + 0.5
        correction = (1 - 0.5 / r) ** 2
        circ = (4 * np.pi * area / (perim ** 2)) * correction
        
        circ_values.append(circ)
    
   
    
    return np.array(circ_values)


def NucD(Ncoord, diag, mask):
    
    """
    Compute normalized distances of Rab spots from the nucleus centroid.

    Parameters:
    - Ncoord: (row, col) — centroid coordinates of the nucleus
    - diag: diagonal length of the enclosing bounding box of the cell
    - mask: 2D labeled image — Rab spot mask within the current cell

    Returns:
    - D: np.array of normalized distances
    """
    D = []

    # Extract Rab cluster centroids
    spot_props = regionprops(mask)
    
    for prop in spot_props:
        
        centroid = prop.centroid  # (row, col) -> (x, y)

        # Check for NaNs
        if not (np.isnan(centroid[0]) or np.isnan(centroid[1])):
            
            # Euclidean distance to nucleus centroid
            d = np.sqrt((centroid[0] - Ncoord[0])**2 + (centroid[1] - Ncoord[1])**2) / diag
            D.append(d)

    return np.array(D)


def NucCd_feret(Ncoord, cell_mask, spot_mask):
    
    """
    Compute normalized distances of Rab spots from the nucleus centroid.
    Distances are normalized by the Feret's diameter.

    Parameters:
    - Ncoord: (row, col) — centroid coordinates of the nucleus
    - spot_mask: 2D labeled image — Rab spot mask within the current cell
    - cell_mask: 2D labeled image — cell mask of the cell of interest containing
                 the spots selected

    Returns:
    - D: np.array of normalized distances
    """
    
    # Find all contours of the cell
    contours = measure.find_contours(cell_mask, level=0.5)
    
    # Select the longest contour, that is the cell
    contour = max(contours, key=len)

    # Compute the convex hull (the smallest polygon that enclose the cell) 
    hull = ConvexHull(contour)
    hull_points = contour[hull.vertices]

    # Compute all pairwise distances between convex hull points.
    # Feret's diameter is the maximum of these distances
    dists = distance.pdist(hull_points)
    Feret_d = np.max(dists)
    
    D = []

    # Extract Rab cluster centroids
    spot_props = regionprops(spot_mask)
    
    for prop in spot_props:
        
        centroid = prop.centroid  # (row, col) -> (x, y)

        # Check for NaNs
        if not (np.isnan(centroid[0]) or np.isnan(centroid[1])):
            
            # Euclidean distance to nucleus centroid
            d = np.sqrt((centroid[0] - Ncoord[0])**2 + (centroid[1] - Ncoord[1])**2) / Feret_d
            D.append(d)

    return np.array(D)


def NucOutd_feret(nucleus_mask, spot_mask, cell_mask):
    
    """
    Compute normalized distances of Rab spots from the nucleus outlines.
    Distances are normalized by the Feret's diameter.

    Parameters:
    - spot_mask: 2D labeled image — Rab spot mask within the current cell
    - nucleus_mask: 2D labeled image - nucleus mask of the cell of interest.
    - cell_mask: 2D labeled image — cell mask of the cell of interest containing
                 the spots selected

    Returns:
    - D: np.array of normalized distances
    """
    ### FERET'S DIAMETER CALCULATION
    
    # Find all contours of the cell
    contours = measure.find_contours(cell_mask, level=0.5)
    
    # Select the longest contour, that is the cell
    contour = max(contours, key=len)

    # Compute the convex hull (the smallest polygon that enclose the cell) 
    hull = ConvexHull(contour)
    hull_points = contour[hull.vertices]

    # Compute all pairwise distances between convex hull points.
    # Feret's diameter is the maximum of these distances
    dists = distance.pdist(hull_points)
    Feret_d = np.max(dists)
    
    
    ### DEFINE THE NUCLUEUS OUTLINES
    
    # get the nucleus outline
    NucOut = nucleus_mask ^ binary_erosion(nucleus_mask)
      
    # Coordinates of perimeter pixels
    Outline_coords = np.column_stack(np.nonzero(NucOut))  # rows = x, cols = y
    
    
    ### COMPUTE THE DISTANCES
    D = []

    # Extract Rab cluster centroids
    spot_props = regionprops(spot_mask)
    spot_cntrd = [p.centroid for p in spot_props]
    
    # Filter out any NaN centroids
    spot_cntrd = [
        (r, c) for r, c in spot_cntrd
        if not (np.isnan(r) or np.isnan(c))
    ]
    
    for xc, yc in spot_cntrd:
        # Compute all distances from this cluster to all nucleus outline pixels
        distances = np.sqrt((Outline_coords[:, 0] - xc)**2 + (Outline_coords[:, 1] - yc)**2)
        D.append(np.min(distances) / Feret_d)
    
    

    return np.array(D)



def PMd(PMperim, diag, mask):
    
    """
    Compute normalized minimum distance of each Rab spot centroid
    from the cell's plasma membrane (cell perimeter).

    Parameters:
    - PMperim: binary image of cell perimeter (True at perimeter pixels)
    - diag: float, diagonal of cell bounding box
    - mask: labeled image of Rab clusters

    Returns:
    - D: np.array of normalized distances (one per cluster)
    """
    # Get centroids of Rab clusters
    spot_props = regionprops(mask)
    spot_cntrd = [p.centroid for p in spot_props]

    # Filter out any NaN centroids
    spot_cntrd = [
        (r, c) for r, c in spot_cntrd
        if not (np.isnan(r) or np.isnan(c))
    ]

    # Coordinates of perimeter pixels
    perim_coords = np.column_stack(np.nonzero(PMperim))  # rows = x, cols = y

    D = []

    for xc, yc in spot_cntrd:
        # Compute all distances from this cluster to all PM pixels
        distances = np.sqrt((perim_coords[:, 0] - xc)**2 + (perim_coords[:, 1] - yc)**2)
        D.append(np.min(distances) / diag)

    return np.array(D)


def PMd_feret(PMperim, spot_mask, cell_mask):
    
    """
    Compute normalized minimum distance of each Rab spot centroid
    from the cell's plasma membrane (cell perimeter). Distances are normalized
    by the Feret's diameter

    Parameters:
    - PMperim: binary image of cell perimeter (True at perimeter pixels)
    - cell_mask: 2D labeled image — cell mask of the cell of interest containing
                 the spots selected
    - spot_mask: labeled image of Rab clusters

    Returns:
    - D: np.array of normalized distances (one per cluster)
    """
    
    # Find all contours of the cell
    contours = measure.find_contours(cell_mask, level=0.5)
    
    # Select the longest contour, that is the cell
    contour = max(contours, key=len)

    # Compute the convex hull (the smallest polygon that enclose the cell) 
    hull = ConvexHull(contour)
    hull_points = contour[hull.vertices]

    # Compute all pairwise distances between convex hull points.
    # Feret's diameter is the maximum of these distances
    dists = distance.pdist(hull_points)
    Feret_d = np.max(dists)
    
    # Get centroids of Rab clusters
    spot_props = regionprops(spot_mask)
    spot_cntrd = [p.centroid for p in spot_props]

    # Filter out any NaN centroids
    spot_cntrd = [
        (r, c) for r, c in spot_cntrd
        if not (np.isnan(r) or np.isnan(c))
    ]

    # Coordinates of perimeter pixels
    perim_coords = np.column_stack(np.nonzero(PMperim))  # rows = x, cols = y

    D = []

    for xc, yc in spot_cntrd:
        # Compute all distances from this cluster to all PM pixels
        distances = np.sqrt((perim_coords[:, 0] - xc)**2 + (perim_coords[:, 1] - yc)**2)
        D.append(np.min(distances) / Feret_d)

    return np.array(D)




def DNN(diag, mask):
    """
    Compute normalized distance to the nearest neighbor (NN)
    for each Rab spot in a cell.

    Parameters:
    - diag: float, diagonal of the bounding box of the cell
    - mask: labeled image of Rab spots

    Returns:
    - D: np.array of NN distances normalized by diag
    """
    # Get centroids of Rab clusters
    spot_props = regionprops(mask)
    centroids = np.array([
        p.centroid for p in spot_props
        if not np.any(np.isnan(p.centroid))
    ])

    if len(centroids) < 2:
        print('No valid ditances to compute. Number of spot < 2')
        return np.array([])  # No valid distances to compute

    #centroids = np.array(centroids)

    # Compute pairwise distances between all centroids
    dist_matrix = cdist(centroids, centroids)

    # Set diagonal (self-distance) to inf to exclude it from min()
    np.fill_diagonal(dist_matrix, np.inf)

    # Get minimum distance for each cluster, normalize by diag
    D = np.min(dist_matrix, axis=1) / diag

    return D

def DNN_feret(cell_mask, spot_mask):
    """
    Compute normalized distance to the nearest neighbor (NN)
    for each Rab spot in a cell. Differences are normalized by the Feret's diameter

    Parameters:
    - cell_mask: 2D labeled image — cell mask of the cell of interest containing
                 the spots selected
    - spot_mask: labeled image of Rab clusters

    Returns:
    - D: np.array of NN distances normalized by diag
    """
    # Find all contours of the cell
    contours = measure.find_contours(cell_mask, level=0.5)
    
    # Select the longest contour, that is the cell
    contour = max(contours, key=len)

    # Compute the convex hull (the smallest polygon that enclose the cell) 
    hull = ConvexHull(contour)
    hull_points = contour[hull.vertices]

    # Compute all pairwise distances between convex hull points.
    # Feret's diameter is the maximum of these distances
    dists = distance.pdist(hull_points)
    Feret_d = np.max(dists)
    
    # Get centroids of Rab clusters
    spot_props = regionprops(spot_mask)
    centroids = np.array([
        p.centroid for p in spot_props
        if not np.any(np.isnan(p.centroid))
    ])

    if len(centroids) < 2:
        print('No valid ditances to compute. Number of spot < 2')
        return np.array([])  # No valid distances to compute

    #centroids = np.array(centroids)

    # Compute pairwise distances between all centroids
    dist_matrix = cdist(centroids, centroids)

    # Set diagonal (self-distance) to inf to exclude it from min()
    np.fill_diagonal(dist_matrix, np.inf)

    # Get minimum distance for each cluster, normalize by diag
    D = np.min(dist_matrix, axis=1) / Feret_d

    return D




def DistSpot(diag, mask):
    """
    Compute the most frequent normalized distance (mode) between Rab spots.

    Parameters:
    - diag: float, diagonal of the bounding box of the cell
    - mask: labeled image of Rab spots

    Returns:
    - D: np.array of mode distances (normalized by diag)
    """
    # Get Rab cluster centroids
    spot_props = regionprops(mask)
    centroids = [
        p.centroid for p in spot_props
        if not np.any(np.isnan(p.centroid))
    ]

    if len(centroids) < 2:
        print("Too few Rab spots to compute distances.")
        return np.array([])

    centroids = np.array(centroids)

    # Compute pairwise distances
    dist_matrix = cdist(centroids, centroids) / diag
    np.fill_diagonal(dist_matrix, np.nan)  # Ignore self-distances


    # get the most frequent distance between all the Rab spots by computing the 
    # mode of all distances
    D = []
    for i in range(len(centroids)):
        distances = dist_matrix[i][~np.isnan(dist_matrix[i])]
        if len(distances) == 0:
            D.append(np.nan)
        else:
            # Floating-point distances are rarely repeated exactly, so mode() 
            # might return strange results. Therefore, it is better to round 
            # the distances before computing the mode
            rounded = np.round(distances, decimals=3)
            mode_val = mode(rounded, keepdims=False).mode
            D.append(mode_val)

    return np.array(D)


def NumNeighb(d, mask, cell):
    
    """
    Count number of Rab spots within distance 'd' of each spot,
    adjusted for how much of the search area lies inside the cell. 
    (Edge's correction')

    Parameters:
    - d: float, radius to search for neighboring clusters (in pixels)
    - mask: labeled image of Rab spots
    - cell: image mask of the cell (nonzero pixels = cell area)

    Returns:
    - num: np.array of adjusted neighbor counts (1 per Rab spot)
    """

    # Binary version of the cell mask
    cell_bw = cell > 0

    # Get centroids of Rab spots
    props = regionprops(mask)
    centroids = [p.centroid for p in props if not np.any(np.isnan(p.centroid))]

   

    centroids = np.array(centroids)

    # Compute all pairwise distances
    dist_matrix = cdist(centroids, centroids)
    np.fill_diagonal(dist_matrix, np.inf)  # Avoid self-counting

    # Initialize result
    num = []
    

    for i, center in enumerate(centroids):
        
        #i = 0
        # Count neighbors within distance d
        neighbors_within_d = np.sum(dist_matrix[i] < d)

        # Build a binary circular mask centered at centroid[i]
        
        # generate zero-matrix where build the circular mask
        matrix = np.zeros_like(cell, dtype=np.uint8)
        
        # generate a grid contianing the coordinates of all the pixels of matrix
        col, row = np.meshgrid(np.arange(cell.shape[1]), np.arange(cell.shape[0]))
        
        # compute the distance of all the fixells falling in the circle centred in the Rab spot
        dist_from_center = np.sqrt((row - center[0])**2 + (col - center[1])**2)
        
        # generate the circular mask
        matrix[dist_from_center <= d] = 1

        # Intersection of search area (circle) and cell
        fraction = matrix * cell_bw

        sumA = np.sum(matrix)      # area of search region
        sumB = np.sum(fraction)    # area of overlap with cell

        # Avoid divide-by-zero
        if sumB == 0:
            weight = 0
        else:
            weight = sumA / sumB  # same logic as your original

        # Weighted neighbor count
        num.append(neighbors_within_d * weight)

    return np.array(num)





















