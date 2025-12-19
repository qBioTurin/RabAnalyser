"""
UMAP + Resolution Scanning Wrapper for R Integration

This script performs:
1. UMAP embedding (2D projection)
2. Resolution scanning to identify optimal gamma for specific number of clusters

Returns dataframes to R for visualization and gamma selection.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import umap
import igraph as ig
import leidenalg
import sys
import os


def build_umap_graph(data, n_neighbors=15, min_dist=0.1, metric="euclidean", random_state=42):
    """
    Build UMAP embedding and extract fuzzy simplicial set graph.
    
    Parameters:
    -----------
    data : np.ndarray
        Scaled feature matrix (N x D)
    n_neighbors : int
        Number of neighbors for UMAP graph
    min_dist : float
        Minimum distance parameter for UMAP
    metric : str
        Distance metric
    random_state : int
        Random seed
    
    Returns:
    --------
    dict with keys:
        - embedding: np.ndarray (N x 2) UMAP coordinates
        - graph: igraph.Graph object with edge weights
        - adjacency: scipy.sparse.csr_matrix (symmetrized)
    """
    
    um = umap.UMAP(
        n_neighbors=n_neighbors,
        n_components=2,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
        set_op_mix_ratio=1.0
    )
    
    embedding = um.fit_transform(data)
    W = um.graph_  # Fuzzy simplicial set (scipy.sparse)
    
    # Ensure symmetry
    W = W.maximum(W.T).tocsr()
    
    # Convert to igraph
    coo = W.tocoo()
    mask = coo.row < coo.col
    edges = list(zip(coo.row[mask], coo.col[mask]))
    weights = coo.data[mask]
    
    g = ig.Graph(n=W.shape[0], edges=edges, directed=False)
    g.es["weight"] = weights.tolist()
    
    return {
        "embedding": embedding,
        "graph": g,
        "adjacency": W
    }


def scan_resolutions(graph, gammas, seed=0):
    """
    Scan resolution parameters to find optimal cluster count.
    
    Parameters:
    -----------
    graph : igraph.Graph
        Weighted UMAP graph
    gammas : np.ndarray
        Array of resolution parameters to test
    seed : int
        Random seed
    
    Returns:
    --------
    pd.DataFrame with columns: gamma, n_clusters
    """
    n_clusters = []
    
    for i, gamma in enumerate(gammas):
        print(f"\rScanning resolution: {100 * (i+1) / len(gammas):.1f}%", end="")
        part = leidenalg.find_partition(
            graph,
            leidenalg.RBConfigurationVertexPartition,
            weights="weight",
            resolution_parameter=gamma,
            seed=seed
        )
        n_clusters.append(len(set(part.membership)))
    
    print()
    return pd.DataFrame({"gamma": gammas, "n_clusters": n_clusters})


def save_graph(graph, adjacency, output_path):
    """
    Save UMAP graph for later use with Leiden clustering.
    
    Parameters:
    -----------
    graph : igraph.Graph
        UMAP graph
    adjacency : scipy.sparse.csr_matrix
        Adjacency matrix
    output_path : str
        Path to save graph
    """
    import pickle
    with open(output_path, 'wb') as f:
        pickle.dump({
            'graph': graph,
            'adjacency': adjacency
        }, f)


def main(data_path,
         n_neighbors=15,
         min_dist=0.1,
         gamma_min=0.1,
         gamma_max=1.0,
         n_gamma_steps=100,
         save_graph_flag=True):
    """
    Main workflow: UMAP → Resolution Scan
    
    Parameters:
    -----------
    data_path : str
        Path to input Excel/CSV file (last column = condition labels)
    n_neighbors : int
        UMAP n_neighbors parameter
    min_dist : float
        UMAP min_dist parameter
    gamma_min : float
        Minimum gamma to scan
    gamma_max : float
        Maximum gamma to scan
    n_gamma_steps : int
        Number of gamma values to test
    save_graph_flag : bool
        If True, save graph for later Leiden clustering
    
    Returns:
    --------
    dict with keys:
        - umap_df: pd.DataFrame with UMAP coordinates and condition labels
        - resolution_scan: pd.DataFrame with gamma and n_clusters
    """
    
    print("Loading data...")
    df = pd.read_csv(data_path)

    # Assume last column is condition labels
    condition_labels = df.iloc[:, -1]
    reduced_df = df.iloc[:, :-1]
        
    # Standardize the data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(reduced_df)
    
    # Initialize UMAP
    print("Building UMAP graph...")
    umap_result = build_umap_graph(data_scaled, n_neighbors=n_neighbors, min_dist=min_dist)
    
    # Build UMAP output dataframe
    umap_df = pd.DataFrame({
        "UMAP1": umap_result["embedding"][:, 0],
        "UMAP2": umap_result["embedding"][:, 1],
        "Class": condition_labels.values
    })
    
    # Resolution scan
    print("Scanning resolution parameters...")
    gammas = np.geomspace(gamma_min, gamma_max, n_gamma_steps)
    res_scan = scan_resolutions(umap_result["graph"], gammas)
    
    # Save graph for later use
    if save_graph_flag:
        output_dir = os.path.dirname(data_path)
        base_name = os.path.splitext(os.path.basename(data_path))[0]
        graph_path = os.path.join(output_dir, f"{base_name}_umap_graph.pkl")
        save_graph(umap_result["graph"], umap_result["adjacency"], graph_path)
        print(f"Graph saved to: {graph_path}")
    
    results = {
        "umap_df": umap_df,
        "resolution_scan": res_scan
    }
    
    print("Done!")
    return results


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python umap_resolution_scan.py <data_path> [n_neighbors] [min_dist] [gamma_min] [gamma_max] [n_gamma_steps]")
        sys.exit(1)
    
    data_path = sys.argv[1]
    n_neighbors = int(sys.argv[2]) if len(sys.argv) > 2 else 15
    min_dist = float(sys.argv[3]) if len(sys.argv) > 3 else 0.1
    gamma_min = float(sys.argv[4]) if len(sys.argv) > 4 else 0.1
    gamma_max = float(sys.argv[5]) if len(sys.argv) > 5 else 1.0
    n_gamma_steps = int(sys.argv[6]) if len(sys.argv) > 6 else 100
    
    results = main(
        data_path=data_path,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        gamma_min=gamma_min,
        gamma_max=gamma_max,
        n_gamma_steps=n_gamma_steps
    )
    
    # Save results as CSV files
    output_dir = os.path.dirname(data_path)
    base_name = os.path.splitext(os.path.basename(data_path))[0]
    
    results["umap_df"].to_csv(os.path.join(output_dir, f"{base_name}_umap.csv"), index=False)
    results["resolution_scan"].to_csv(os.path.join(output_dir, f"{base_name}_resolution_scan.csv"), index=False)
    
    print(f"\nResults saved to: {output_dir}")
