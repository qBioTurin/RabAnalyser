"""
UMAP + Leiden Clustering Wrapper for R Integration

This script performs:
1. UMAP embedding (2D projection)
2. Leiden clustering on UMAP graph
3. Resolution scanning to find optimal cluster count
4. Cluster stability analysis via bootstrap subsampling

Returns dataframes to R for visualization.
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
    
    for gamma in gammas:
        part = leidenalg.find_partition(
            graph,
            leidenalg.RBConfigurationVertexPartition,
            weights="weight",
            resolution_parameter=gamma,
            seed=seed
        )
        n_clusters.append(len(set(part.membership)))
        

    return pd.DataFrame({"gamma": gammas, "n_clusters": n_clusters})


def run_leiden(graph, resolution=0.42, seed=0):
    """
    Run Leiden clustering on UMAP graph.
    
    Parameters:
    -----------
    graph : igraph.Graph
        Weighted UMAP graph
    resolution : float
        Resolution parameter (higher = more clusters)
    seed : int
        Random seed
    
    Returns:
    --------
    np.ndarray
        Cluster labels (0-indexed)
    """
    part = leidenalg.find_partition(
        graph,
        leidenalg.RBConfigurationVertexPartition,
        weights="weight",
        resolution_parameter=resolution,
        seed=seed
    )
    return np.array(part.membership)


def cluster_stability(adjacency, ref_labels, resolution=0.42, n_bootstrap=100, subsample_prop=0.8, seed=42):
    """
    Measure cluster stability via bootstrap subsampling.
    
    Parameters:
    -----------
    adjacency : scipy.sparse.csr_matrix
        Symmetrized UMAP graph adjacency
    ref_labels : np.ndarray
        Reference cluster labels from full dataset
    resolution : float
        Resolution parameter used for clustering
    n_bootstrap : int
        Number of bootstrap iterations
    subsample_prop : float
        Proportion of samples to keep in each bootstrap
    seed : int
        Random seed
    
    Returns:
    --------
    pd.DataFrame with columns: bootstrap, cluster, jaccard
    """
    N = adjacency.shape[0]
    rng = np.random.default_rng(seed)
    
    results = []
    
    for b in range(n_bootstrap+1):
        print(f"\rStability analysis: {100 * (b) / n_bootstrap:.1f}%", end="")
        
        # Subsample
        sub_ids = np.sort(rng.choice(N, size=int(subsample_prop * N), replace=False))
        
        # Induce subgraph
        sub = adjacency[sub_ids][:, sub_ids].tocoo()
        edges_sub = list(zip(sub.row.tolist(), sub.col.tolist()))
        weights_sub = sub.data.tolist()
        
        g_sub = ig.Graph(n=len(sub_ids), edges=edges_sub, directed=False)
        g_sub.es["weight"] = weights_sub
        
        # Recluster
        part_sub = leidenalg.find_partition(
            g_sub,
            leidenalg.RBConfigurationVertexPartition,
            weights="weight",
            resolution_parameter=resolution,
            seed=0 + b
        )
        
        labels_sub = np.array(part_sub.membership, dtype=int)
        ref_on_sub = ref_labels[sub_ids]
        
        # Compact subsample labels
        _, sub_compact = np.unique(labels_sub, return_inverse=True)
        
        # Contingency matrix
        n_ref = ref_on_sub.max() + 1
        n_sub = sub_compact.max() + 1
        M = np.zeros((n_ref, n_sub), dtype=int)
        np.add.at(M, (ref_on_sub, sub_compact), 1)
        
        # Jaccard index
        n_i = M.sum(axis=1, keepdims=True)
        n_j = M.sum(axis=0, keepdims=True)
        denom = n_i + n_j - M
        with np.errstate(divide='ignore', invalid='ignore'):
            J = np.where(denom > 0, M / denom, 0.0)
        
        best_J = J.max(axis=1)
        
        # Store results
        for cluster_id, jaccard in enumerate(best_J):
            results.append({
                "bootstrap": b,
                "cluster": cluster_id,
                "jaccard": jaccard
            })
    
    print()
    return pd.DataFrame(results)


def main(data_path,
         n_neighbors=15, # user can change it
         min_dist=0.1,
         resolution=0.42, # user can change it
         n_bootstrap=100,
         subsample_prop=0.8,
         scan_resolutions_flag=True,
         stability_analysis_flag=True):
    """
    Main workflow: UMAP → Leiden → stability analysis
    
    Parameters:
    -----------
    data_path : str
        Path to input Excel/CSV file (last column = condition labels)
    n_neighbors : int
        UMAP n_neighbors parameter
    min_dist : float
        UMAP min_dist parameter
    resolution : float
        Leiden resolution parameter
    n_bootstrap : int
        Number of bootstrap runs for stability
    subsample_prop : float
        Subsample proportion for stability
    scan_resolutions_flag : bool
        If True, scan resolution parameters
    stability_analysis_flag : bool
        If True, perform stability analysis
    
    Returns:
    --------
    dict with keys:
        - umap_df: pd.DataFrame with UMAP coordinates and cluster labels
        - resolution_scan: pd.DataFrame (if scan_resolutions_flag=True)
        - stability: pd.DataFrame (if stability_analysis_flag=True)
    """
    
    print("Loading data...")
    df = pd.read_csv(data_path)


    # Assume last column is condition labels
    condition_labels = df.iloc[:, -1]
    reduced_df = df.iloc[:, :-1]
        
    # Standardize the data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(reduced_df)
    print(data_scaled)
    # Initialize UMAP
    print("Building UMAP graph...")
    umap_result = build_umap_graph(data_scaled, n_neighbors=n_neighbors, min_dist=min_dist)
    
    print("Running Leiden clustering...")
    labels = run_leiden(umap_result["graph"], resolution=resolution)
    
    # Build output dataframe
    umap_df = pd.DataFrame({
        "UMAP1": umap_result["embedding"][:, 0],
        "UMAP2": umap_result["embedding"][:, 1],
        "Cluster": labels,
        "Class": condition_labels.values
    })
    
    results = {
        "umap_df": umap_df
    }
    
    # Optional: resolution scan
    if scan_resolutions_flag:
        print("Scanning resolution parameters...")
        gammas = np.geomspace(0.1, 1, 100)
        res_scan = scan_resolutions(umap_result["graph"], gammas)
        results["resolution_scan"] = res_scan
    
    # Optional: stability analysis
    if stability_analysis_flag:
        print("Running stability analysis...")
        stability_df = cluster_stability(
            umap_result["adjacency"],
            labels,
            resolution=resolution,
            n_bootstrap=n_bootstrap,
            subsample_prop=subsample_prop
        )
        results["stability"] = stability_df
    
    print("Done!")
    return results


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python umap_leiden_wrapper.py <data_path> [n_neighbors] [min_dist] [resolution] [n_bootstrap]")
        sys.exit(1)
    
    data_path = sys.argv[1]
    n_neighbors = int(sys.argv[2]) if len(sys.argv) > 2 else 15
    min_dist = float(sys.argv[3]) if len(sys.argv) > 3 else 0.1
    resolution = float(sys.argv[4]) if len(sys.argv) > 4 else 0.42
    n_bootstrap = int(sys.argv[5]) if len(sys.argv) > 5 else 100
    
    results = main(
        data_path=data_path,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        resolution=resolution,
        n_bootstrap=n_bootstrap
    )
    
    # Save results as CSV files
    output_dir = os.path.dirname(data_path)
    base_name = os.path.splitext(os.path.basename(data_path))[0]
    
    results["umap_df"].to_csv(os.path.join(output_dir, f"{base_name}_umap_clusters.csv"), index=False)
    
    if "resolution_scan" in results:
        results["resolution_scan"].to_csv(os.path.join(output_dir, f"{base_name}_resolution_scan.csv"), index=False)
    
    if "stability" in results:
        results["stability"].to_csv(os.path.join(output_dir, f"{base_name}_stability.csv"), index=False)
    
    print(f"\nResults saved to: {output_dir}")
