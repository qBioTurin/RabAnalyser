"""
Leiden Clustering Wrapper for R Integration

This script performs:
1. Load pre-computed UMAP graph
2. Leiden clustering with specified gamma (resolution parameter)
3. Cluster stability analysis via bootstrap subsampling

Returns dataframes to R for visualization.
"""

import pandas as pd
import numpy as np
import igraph as ig
import leidenalg
import sys
import os
import pickle


def load_graph(graph_path):
    """
    Load pre-computed UMAP graph.
    
    Parameters:
    -----------
    graph_path : str
        Path to saved graph pickle file
    
    Returns:
    --------
    dict with keys:
        - graph: igraph.Graph object
        - adjacency: scipy.sparse.csr_matrix
    """
    with open(graph_path, 'rb') as f:
        data = pickle.load(f)
    return data


def run_leiden(graph, resolution=0.42, seed=0):
    """
    Run Leiden clustering on UMAP graph.
    
    Parameters:
    -----------
    graph : igraph.Graph
        Weighted UMAP graph
    resolution : float
        Resolution parameter (gamma) - higher = more clusters
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
    
    for b in range(n_bootstrap + 1):
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


def main(umap_csv_path,
         graph_path,
         resolution=0.42,
         n_bootstrap=100,
         subsample_prop=0.8,
         stability_analysis_flag=True):
    """
    Main workflow: Leiden Clustering → Stability Analysis
    
    Parameters:
    -----------
    umap_csv_path : str
        Path to UMAP coordinates CSV file (from umap_resolution_scan.py)
    graph_path : str
        Path to saved UMAP graph pickle file
    resolution : float
        Leiden resolution parameter (gamma)
    n_bootstrap : int
        Number of bootstrap runs for stability
    subsample_prop : float
        Subsample proportion for stability
    stability_analysis_flag : bool
        If True, perform stability analysis
    
    Returns:
    --------
    dict with keys:
        - umap_df: pd.DataFrame with UMAP coordinates and cluster labels
        - stability: pd.DataFrame (if stability_analysis_flag=True)
    """
    
    print("Loading UMAP coordinates...")
    umap_df = pd.read_csv(umap_csv_path)
    
    print("Loading UMAP graph...")
    graph_data = load_graph(graph_path)
    
    print(f"Running Leiden clustering with gamma={resolution}...")
    labels = run_leiden(graph_data["graph"], resolution=resolution)
    
    # Add cluster labels to UMAP dataframe
    umap_df["Cluster"] = labels
    
    results = {
        "umap_df": umap_df
    }
    
    # Optional: stability analysis
    if stability_analysis_flag:
        print("Running stability analysis...")
        stability_df = cluster_stability(
            graph_data["adjacency"],
            labels,
            resolution=resolution,
            n_bootstrap=n_bootstrap,
            subsample_prop=subsample_prop
        )
        results["stability"] = stability_df
    
    print("Done!")
    return results


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python leiden_clustering.py <umap_csv_path> <graph_path> [resolution] [n_bootstrap] [subsample_prop]")
        sys.exit(1)
    
    umap_csv_path = sys.argv[1]
    graph_path = sys.argv[2]
    resolution = float(sys.argv[3]) if len(sys.argv) > 3 else 0.42
    n_bootstrap = int(sys.argv[4]) if len(sys.argv) > 4 else 100
    subsample_prop = float(sys.argv[5]) if len(sys.argv) > 5 else 0.8
    
    results = main(
        umap_csv_path=umap_csv_path,
        graph_path=graph_path,
        resolution=resolution,
        n_bootstrap=n_bootstrap,
        subsample_prop=subsample_prop
    )
    
    # Save results as CSV files
    output_dir = os.path.dirname(umap_csv_path)
    base_name = os.path.splitext(os.path.basename(umap_csv_path))[0].replace("_umap", "")
    
    results["umap_df"].to_csv(os.path.join(output_dir, f"{base_name}_leiden_clusters.csv"), index=False)
    
    if "stability" in results:
        results["stability"].to_csv(os.path.join(output_dir, f"{base_name}_stability.csv"), index=False)
    
    print(f"\nResults saved to: {output_dir}")
