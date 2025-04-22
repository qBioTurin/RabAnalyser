import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import MyFunctions2 as F 
from sklearn.cluster import KMeans
import seaborn as sns
import umap
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelBinarizer
from itertools import combinations
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
import sys
from sklearn.model_selection import train_test_split

from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

import igraph as ig
import leidenalg
from sklearn.metrics import silhouette_score, davies_bouldin_score

import matplotlib.patches as mpatches

import matplotlib.colors as mcolors ###
import matplotlib.cm as cm ###
from statsmodels.stats.proportion import proportions_ztest ###

"""
This script analyse the data matrix obtained by the
'SC_KS_singlePopulation' script to perform subpopulation 
identification and characterization. Here, both control and treatment 
are merged in an unique data matrix to allow direct comparison between the 
conditions.

input: KS matrix

output:
    
Feature correlation matrices
UMAP plot
UMAP plot with clusters
Feature values visualization in UMAP

 

"""

""" File loading """
print("file loading")
# Specify the folder containing the .mat files
#Data_path = 'C:/Users/E14Ge/Desktop/NewData il mio paper/dati singole cellule e farmaci KS/rab11/All.xlsx'
Data_path = sys.argv[1] # 'Data_path = 'C:/Users/E14Ge/Desktop/NewData il mio paper/dati singole cellule e farmaci KS/rab11/Nocodazole_SCrab11_withControlRef2.xlsx'

# Load the Excel file into a DataFrame
df = pd.read_excel(Data_path)

# Remove the column with treatment labels
df_selected = df.iloc[:, :-1]



""" FEATURE SELECTION """
print("data filtering")

#here it is performed the feature lection step

# values pearson coefficient to be used as threshold
threshold = 0.75 
reduced_df = F.FilterFeat(df_selected, threshold)


### CORRELATION MATRIX PLOT ###

# compute the correlation matrix of the original features
correlation_matrixOLD = df_selected.corr()

# Plot the correlation matrix of the original features
plt.figure(figsize=(12, 10))
sns.heatmap(
    correlation_matrixOLD,
    annot=False,      # Set to True if you want to show correlation values
    cmap='coolwarm',  # Colormap for the heatmap
    xticklabels=correlation_matrixOLD.columns,  # Use feature names for x-axis
    yticklabels=correlation_matrixOLD.columns,  # Use feature names for y-axis
    cbar=True,              # Show color bar
    vmax=1,
    vmin=-1
)

plt.title('Correlation Matrix of original Features', fontsize=16)
plt.xticks(rotation=90, fontsize=8)  # Rotate x-axis labels for readability
plt.yticks(fontsize=8)               # Adjust y-axis label font size
plt.tight_layout()

#plt.savefig('D:/PhD Terzo anno/TESI/immagini/CorrMatrixRab11Allold.svg',transparent=True)
#plt.savefig('C:/Users/E14Ge/Desktop/Data_results/allTRTDRab5UMAP.pdf')
#plt.savefig('C:/Users/E14Ge/Desktop/Data_results/allTRTDRab5UMAP.png',dpi=600)

plt.show()


# compute the correlation matrix of the original features
correlation_matrixNEW = reduced_df.corr()


# Plot the correlation matrix of the filtered features
plt.figure(figsize=(12, 10))
sns.heatmap(
    correlation_matrixNEW,
    annot=False,      # Set to True if you want to show correlation values
    cmap='coolwarm',  # Colormap for the heatmap
    xticklabels=correlation_matrixNEW.columns,  # Use feature names for x-axis
    yticklabels=correlation_matrixNEW.columns,  # Use feature names for y-axis
    cbar=True,              # Show color bar
    vmax=1,
    vmin=-1
)

plt.title('Correlation Matrix of filterd Features', fontsize=16)
plt.xticks(rotation=90, fontsize=8)  # Rotate x-axis labels for readability
plt.yticks(fontsize=8)               # Adjust y-axis label font size
plt.tight_layout()

#plt.savefig('D:/PhD Terzo anno/TESI/immagini/CorrMatrixRab11AllNEW.svg',transparent=True)
#plt.savefig('C:/Users/E14Ge/Desktop/Data_results/allTRTDRab5UMAP.pdf')
#plt.savefig('C:/Users/E14Ge/Desktop/Data_results/allTRTDRab5UMAP.png',dpi=600)

plt.show()


""" UMAP ANALYSIS """


# Standardize the data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(reduced_df)


# Initialize UMAP
reducer = umap.UMAP(n_neighbors=10, min_dist=0.5, n_components=2, random_state=42)

# Fit and transform the data
Data_UMAP = reducer.fit_transform(data_scaled)

#create a dataframe with the umap dimension as columns
umap_df = pd.DataFrame(Data_UMAP, columns=["UMAP1", "UMAP2"])


""" UMAP PLOT """

# UMAP plot of control and treated cells

# extract conditions labels 
TRTD_label = df['Treatment']

# add conditions labels to UMAP dataframe
umap_df = pd.DataFrame(Data_UMAP, columns=["UMAP1", "UMAP2"])
umap_df["Treatment"] = TRTD_label.values


control = 'DMSO'
treatment= 'Nocodazole'

# Define RGB colors (normalized between 0 and 1)
color_dict = {
    control: (0.6, 0.6, 0.6),   # gray
    treatment: (0.94, 0.54, 0.38)  # red
}

# plot
plt.figure(figsize=(8, 6))
for treatment in umap_df["Treatment"].unique():
    subset = umap_df[umap_df["Treatment"] == treatment]
    plt.scatter(
        subset["UMAP1"], subset["UMAP2"], label=f"{treatment}", s = 30, alpha=0.7,
        c=[color_dict[treatment]]
    )
plt.title("UMAP Visualization by Treatment")
plt.xlabel("UMAP1")
plt.ylabel("UMAP2")
plt.legend()
plt.grid(False)

#plt.savefig('D:/PhD Terzo anno/TESI/immagini/allRab11proportion_CNDTN.svg',transparent=True)
#plt.savefig('C:/Users/E14Ge/Desktop/Data_results/allTRTDRab5UMAP.pdf')
#plt.savefig('C:/Users/E14Ge/Desktop/Data_results/allTRTDRab5UMAP.png',dpi=600)

plt.show()


""" CLUSTERING """

# Here I perform clustering by KMeans, Leiden and DBSCAN. Users can decide
# the algorithm more appropriated. Morevoer, to unbiasly decide
# the number of clusters I get help from several scores/indeces such as:
# Silhouette Score, elbow method, DBI score and modularity score. 


# check as as True the algorithm that you want to use
K_mean_check = True
DBSCAN_check = False
leiden_check = False


### Kmeans clustering ###

if K_mean_check:

    # Elbow method
    
    # Calculate inertia for a range of k values
    inertia = []
    k_range = range(1, 11)  # Test from 1 to 10 clusters
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(Data_UMAP)
        inertia.append(kmeans.inertia_)
    
    
    # Plot the elbow curve
    
    plt.figure(figsize=(8, 6))
    
    plt.plot(k_range, inertia, marker='o')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia')
    plt.title('Elbow Method')
    plt.show()
    
    
    ### SILHOUETTE SCORE ###
    
    silhouette_scores = []
    k_range = range(2, 11) # Silhouette requires at least 2 clusters
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(Data_UMAP)
        score = silhouette_score(Data_UMAP, labels)
        silhouette_scores.append(score)
    
    
    # Plot silhouette scores
    plt.figure(figsize=(8, 6))
    
    plt.plot(k_range, silhouette_scores, marker='o')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Analysis')
    plt.show()
    
    # Optimal k is where the silhouette score is maximized
    optimal_k = k_range[silhouette_scores.index(max(silhouette_scores))]
    print(f"Optimal number of clusters: {optimal_k}")
    
    # I perform KMeans clustering with optiaml number of clusters
    # based on Elbow and Silhouette score. The optimal can be selected by user
    # according to previuos indices.
    
    # Define number of clusters
    optimal = 2
    n_clusters = optimal
    
    # Initialize k-means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    
    # Fit k-means to your data
    kmeans.fit(Data_UMAP)
    
    # Get cluster labels
    labels = kmeans.labels_
    unique_clusters = np.unique(labels)
    
    # PLOTTING
    
    plt.figure(figsize=(8, 6))

    # Use the 'viridis' colormap
    viridis = cm.get_cmap('viridis', n_clusters)
    colors = viridis(np.linspace(0, 1, n_clusters))
    
    for cluster_id in range(n_clusters):
        cluster_points = umap_df[labels == cluster_id]
        plt.scatter(
            cluster_points['UMAP1'], cluster_points['UMAP2'],
            c=[colors[cluster_id]],  # Assign cluster-specific color
            label=f"Cluster {cluster_id + 1}",
            s=20, alpha=0.7
        )
    
    # Add legend
    plt.legend(title="Clusters")
    plt.title('K-Means Clustering on UMAP Data')
    plt.xlabel('UMAP1')
    plt.ylabel('UMAP2')
    
    # Optional: Save figures
    # plt.savefig('D:/PhD Terzo anno/TESI/immagini/allZfyveRab5UMAPclusters.svg', transparent=True)
    # plt.savefig('C:/Users/E14Ge/Desktop/Data_results/ControlRab11UMAP_ref2.svg', transparent=True)
    # plt.savefig('C:/Users/E14Ge/Desktop/Data_results/ControlRab11UMAP_ref2.pdf')
    # plt.savefig('C:/Users/E14Ge/Desktop/Data_results/ControlRab11UMAP_ref2.png', dpi=600)
    
    plt.show()
    
    # add to the cells the labes indicating the belonging to a cluster
    reduced_df['Clusters'] = labels
    
### DBSCAN ###

if DBSCAN_check:
    
         
     # Compute the distance to the k-th nearest neighbor (how far you have to go
     # from each point to reach its k-th closest neighbor) to find your starting 
     # guess for eps paramenter. The elbow represent the starting guess for eps.
     
     
     min_samples = 16 # Usually, min_samples = 2 * num of features
     
     k = 15 #k = min_samples - 1; 
     neighbors = NearestNeighbors(n_neighbors=k)
     neighbors_fit = neighbors.fit(Data_UMAP)
     distances, indices = neighbors_fit.kneighbors(Data_UMAP)
     
     # Sort distances and plot
     distances = np.sort(distances[:, k-1], axis=0)
     plt.plot(distances)
     plt.xlabel("Data Points Sorted by Distance")
     plt.ylabel(f"Distance to {k}-th Nearest Neighbor")
     plt.title("K-Distance Plot for Choosing eps")
     plt.show()
     
     
     # I will use the starting guess for eps value to generate a eps values range
     # to measure silhoutte scores and plot them against eps values
     
     # insert the starting guess for eps value
     elbow_eps = 0.9
     eps_range = np.linspace(elbow_eps * 0.8, elbow_eps * 1.2, 10) 
     
     silhouette_scores = []
     num_clusters_list = []
     
     for eps in eps_range:
         
         db = DBSCAN(eps=eps, min_samples=min_samples).fit(Data_UMAP)
         
         labels = db.labels_
         
         # count the number of clusters returned by DBSCAN, excluding the noise 
         # points.
         n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
     
         # Compute silhouette only if at least 2 clusters
         if n_clusters >= 2:
             score = silhouette_score(Data_UMAP, labels)
         else:
             score = -1
     
         silhouette_scores.append(score)
         num_clusters_list.append(n_clusters)
     
     # Plot silhouette score vs. eps
     plt.figure()
     plt.plot(eps_range, silhouette_scores, marker='o')
     plt.xlabel("eps")
     plt.ylabel("Silhouette score")
     plt.title("Silhouette score vs. eps")
     plt.grid(True)
     plt.show()
    
     # Plot number of clusters vs. eps
     plt.figure()
     plt.plot(eps_range, num_clusters_list, marker='o')
     plt.xlabel("eps")
     plt.ylabel("Number of clusters (excluding noise)")
     plt.title("Number of clusters vs. eps")
     plt.grid(True)
     plt.show()
     
     
     # based on the plotted graphs select the optimal eps value
     optimal_eps = 0.8
     
     
     # Perform DBSCAN clustering
     dbscan = DBSCAN(eps=optimal_eps, min_samples=min_samples)  # Adjust parameters as needed
     labels = dbscan.fit_predict(Data_UMAP)
     
     
     # Identify core cluster labels and noise
     core_mask = labels != -1
     noise_mask = labels == -1
    
     # Normalize cluster labels to be used with viridis colormap
     unique_clusters = np.unique(labels[core_mask])
     n_clusters = len(unique_clusters)
     print(f"Estimated number of clusters: {n_clusters}")
    
     # Create color map
     cmap = plt.cm.get_cmap('viridis', n_clusters)
    
     # Plot cluster points using viridis
     plt.figure(figsize=(8, 6))
     scatter = plt.scatter(
        Data_UMAP[core_mask, 0],Data_UMAP[core_mask, 1],
        c=labels[core_mask],
        cmap=cmap,
        
        label="Cluster points",
        alpha = 0.7
     )
    
     # Plot noise points in black
     plt.scatter(
        Data_UMAP[noise_mask, 0], Data_UMAP[noise_mask, 1],
        c='black',
        
        label="Noise",
        alpha = 0.7
     )
     
           
     plt.title("DBSCAN Clustering")
     plt.xlabel("UMAP1")
     plt.ylabel("UMAP2")
     plt.legend()
     plt.tight_layout()
     
     #plt.savefig('D:/PhD Terzo anno/TESI/immagini/allZfyveRab5UMAPclusters.svg',transparent=True)
     #plt.savefig('C:/Users/E14Ge/Desktop/Data_results/ControlRab11UMAP_ref2.svg',transparent=True)
     #plt.savefig('C:/Users/E14Ge/Desktop/Data_results/ControlRab11UMAP_ref2.pdf')
     #plt.savefig('C:/Users/E14Ge/Desktop/Data_results/ControlRab11UMAP_ref2.png',dpi=600)
     
     plt.show()
     
     # add to the cells the labes indicating the belonging to a cluster
     reduced_df['Clusters'] = labels

### LEIDEN ###

if leiden_check:
    
    # for the Leiden algorithm the number of clusters depends on the resolution
    # parameter. Here, I test several resolution parameters 
    # Define a range of resolution values for Leiden clustering
    resolution_values = np.linspace(0.001, 0.1, 50)  # Testing different granularity levels
    
    silhouette_scores = []
    dbi_scores = []
    modularity_scores = []
    num_clusters = []
    
    # Store UMAP connectivity graph
    umap_graph = reducer.graph_  # Extract UMAP's internal graph
    
    # Create an igraph Graph (assuming you have the UMAP graph)
    edges = np.array(umap_graph.nonzero()).T  # Extract edges
    g = ig.Graph(edges.tolist(), directed=False)
    
    
    # Loop through different resolutions
    for res in resolution_values:
        # Run Leiden clustering at current resolution
        partition = leidenalg.find_partition(g, leidenalg.CPMVertexPartition, resolution_parameter=res, n_iterations=-1,seed=42)
        
        # Get cluster labels
        labels = np.array(partition.membership)
        
        # Compute number of clusters
        num_clusters.append(len(set(labels)))
    
        # Compute Modularity Score
        modularity_scores.append(partition.modularity)
    
        # Compute Silhouette Score (only if more than 1 cluster)
        if len(set(labels)) > 1:
            silhouette_scores.append(silhouette_score(umap_df[["UMAP1", "UMAP2"]], labels))
            dbi_scores.append(davies_bouldin_score(umap_df[["UMAP1", "UMAP2"]], labels))
        else:
            silhouette_scores.append(-1)  # Invalid value if only one cluster
            dbi_scores.append(np.inf)  # Invalid value if only one cluster
    
    # Plot Metrics vs. Number of Clusters
    plt.figure(figsize=(10, 5))
    
   
    plt.plot(resolution_values, silhouette_scores, marker="o", markersize=3, label="Silhouette Score")
    plt.plot(resolution_values, dbi_scores, marker="s", markersize=3, label="DBI Score")
    plt.xlabel("resolutions values (a.u.)")
    plt.ylabel("Score")
    plt.legend()
    
    #plt.savefig('D:/PhD Terzo anno/TESI/immagini/allRab11_LeidenScores.svg',transparent=True)
    
    plt.title("Silhouette & DBI Scores")
    
    # Plot Metrics vs. Number of Clusters
    plt.figure(figsize=(10, 5))
    plt.plot(resolution_values, modularity_scores, marker="o", color="red", markersize=3, label="Modularity Score")
    plt.xlabel("resolutions values (a.u.)")
    plt.ylabel("Modularity Score")
    plt.legend()
    #plt.title("Modularity vs. Cluster Number")
    
    #plt.savefig('D:/PhD Terzo anno/TESI/immagini/allRab11_LeidenScores2.svg',transparent=True)
    
    plt.show()
    
    
    #choose the more appropriate resolution parameter
    res = 0.007
    
    # Store UMAP connectivity graph
    umap_graph = reducer.graph_  # Extract UMAP's internal graph
    
    # Create an igraph Graph (assuming you have the UMAP graph)
    edges = np.array(umap_graph.nonzero()).T  # Extract edges
    g = ig.Graph(edges.tolist(), directed=False)
    
   
    # Perform Leiden clustering with resolution parameter
    partition = leidenalg.find_partition(g, leidenalg.CPMVertexPartition, resolution_parameter=res, n_iterations=-1,seed=42) #0.0048
    
    # Extract cluster labels
    labels = partition.membership
    
    # Create a DataFrame with UMAP dimensions and cluster labels
    umap_df = pd.DataFrame(Data_UMAP, columns=["UMAP1", "UMAP2"])
    umap_df["Leiden_Cluster"] = labels
    
    
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(umap_df["UMAP1"], umap_df["UMAP2"], c=umap_df["Leiden_Cluster"], cmap="viridis", alpha=0.7)
    
    # Create a legend for the clusters

    # Get unique cluster labels
    unique_clusters = np.unique(umap_df["Leiden_Cluster"])
    
    # Create legend handles
    legend_patches = [mpatches.Patch(color=scatter.cmap(scatter.norm(cluster)), label=f"Cluster {cluster}") for cluster in unique_clusters]
    
    # Add legend
    plt.legend(handles=legend_patches, title="Subpopulations", loc="upper right")
    
    # Labels and title
    plt.xlabel("UMAP1")
    plt.ylabel("UMAP2")
    plt.title("Leiden Clustering in UMAP Space")
    
    #plt.savefig('D:/PhD Terzo anno/TESI/immagini/allRab11_UMAPleiden.svg',transparent=True)
    
    plt.show()
    
    # add to the cells the labes indicating the belonging to a cluster
    reduced_df['Clusters'] = labels
    

""" SUBPOPULATION PROPORTION """

# Count occurrences of each unique integer
unique_values, counts = np.unique(labels, return_counts=True)

# Compute proportions
proportions = counts / counts.sum()

# Generate colors from colormap
colors = plt.cm.viridis(np.linspace(0, 1, len(unique_clusters))) 


# Create the stacked bar plot
fig, ax = plt.subplots(figsize=(6, 6))
bottom = 0  # Start stacking from 0

print(enumerate(unique_values))
for idx, value in enumerate(unique_values):
    ax.bar("Proportion", proportions[idx], bottom=bottom, color=colors[idx], label=f"{value}")
    bottom += proportions[idx]  # Update bottom for stacking

# Formatting
ax.set_ylabel("Proportion")
ax.set_yticks(np.arange(0, 1.1, 0.25))
ax.legend(title="Subpopulations", bbox_to_anchor=(1.05, 1), loc='upper left')

#plt.savefig('D:/PhD Terzo anno/TESI/immagini/allRab11proportion_subpop.svg',transparent=True)

# Show the plot
plt.show()


""" CTRL AND TRTD CELLS PROPORTION PER SINGLE SUBPOPULATION """

# add information about conditions of the cells: control or treatment
reduced_df['Condition'] = df['Treatment']

# Count occurrences of each condition within each cluster
cluster_counts = reduced_df.groupby(["Clusters", "Condition"]).size().unstack(fill_value=0)

# Normalize by total cells in each cluster to get fractions
cluster_fractions = cluster_counts.div(cluster_counts.sum(axis=1), axis=0)

# extract the columns name of 'cluster_fractions' dataframe
names = cluster_fractions.columns


# Extract cluster names and fraction values
clusters = cluster_fractions.index
control_fractions = cluster_fractions[names[0]]
treated_fractions = cluster_fractions[names[1]]

# Create stacked bar plot
plt.figure(figsize=(8, 6))
plt.bar(clusters, control_fractions, label="Control", color=(0.6, 0.6, 0.6))
plt.bar(clusters, treated_fractions, bottom=control_fractions, label="Treated", color=(0.94, 0.54, 0.38))

# Labels and title
plt.xlabel("Cluster")
plt.ylabel("Proportion")
plt.title("Fraction of Control and Treated Cells per Cluster")
plt.legend()

#plt.savefig('D:/PhD Terzo anno/TESI/immagini/allRab11proportion_CNDTN.svg',transparent=True)

plt.show()


""" CELL PROPORTION IN CONTROL AND TREATED CONDITION """

# I quantify the proportion of cells for each condition and in each subpopulation.
# Proportions are calculate with respect to the total number of cells belonging
# to each condition

# Count total number of control and treated cells
total_control = (reduced_df['Condition'] == names[0]).sum()
total_treated = (reduced_df['Condition'] == names[1]).sum()

# Count occurrences of each condition within each cluster
cluster_counts = reduced_df.groupby(["Clusters", "Condition"]).size().unstack(fill_value=0)

# Normalize by total control or treated cells in the dataset
cluster_fractions = cluster_counts.copy()
cluster_fractions[names[0]] /= total_control  # Normalize control cells
cluster_fractions[names[1]] /= total_treated  # Normalize treated cells

# Extract cluster names and fraction values
clusters = cluster_fractions.index
control_fractions = cluster_fractions[names[0]]
treated_fractions = cluster_fractions[names[1]]

# Define positions for each cluster
x = np.arange(len(clusters))
bar_width = 0.4

# Create a grouped bar plot for each subpopulation
fig, ax = plt.subplots(figsize=(10, 6))

# Plot bars side by side for Control and Treated
ax.bar(x - bar_width/2, control_fractions, width=bar_width, label="Control", color=(0.6, 0.6, 0.6))
ax.bar(x + bar_width/2, treated_fractions, width=bar_width, label="Treated", color=(0.94, 0.54, 0.38))

# Labels and title
ax.set_xticks(x)
ax.set_xticklabels(clusters, rotation=45)  # Rotate labels for better readability
ax.set_xlabel("Cluster (Subpopulation)")
ax.set_ylabel("Fraction of Total Population")
ax.set_title("Comparison of Cell Fraction Between Control and Treated Per Subpopulation")
ax.legend(title="Condition")

#plt.savefig('D:/PhD Terzo anno/TESI/immagini/allRab11proportion_CNDTNcomp.svg',transparent=True)

plt.show()

# I quantify the statistical significance among the differences of the various 
# proportion by Z-test. I adjusted the p-value bu Bonferroni correction. Results
# are returned as a table.

# Store p-values
p_values = []

# Perform Z-test for each cluster
for cluster in clusters:
    count = np.array([control_fractions[cluster] * total_control, treated_fractions[cluster] * total_treated])
    nobs = np.array([total_control, total_treated])

    stat, p_value = proportions_ztest(count, nobs)
    p_values.append(p_value)
    
    print(f"Subpopulation {cluster}: Z-test p-value = {p_value:.4f}")

# Interpret results
significant_clusters = [clusters[i] for i, p in enumerate(p_values) if p < 0.05]
if significant_clusters:
    print("\nSignificant differences found in subpopulations:", significant_clusters)
else:
    print("\nNo significant differences found between Control and Treated.")
    
    
# Apply Bonferroni correction
adjusted_p = multipletests(p_values, method='bonferroni')[1]

# Print corrected results
for i, cluster in enumerate(clusters):
    print(f"Subpopulation {cluster}: Adjusted p-value = {adjusted_p[i]:.4f}")

# Identify significant clusters after correction
significant_clusters_corrected = [clusters[i] for i, p in enumerate(adjusted_p) if p < 0.05]
if significant_clusters_corrected:
    print("\nSignificant differences (after correction) found in subpopulations:", significant_clusters_corrected)
else:
    print("\nNo significant differences found after correction.")




""" FEATURES VALUES VISUALIZATION IN UMAP """

# I visualize in the UMAP space the features values. I use
# only the features selected after feature selection

# Define a custom colormap transitioning from Blue to White to Green
BWG = mcolors.LinearSegmentedColormap.from_list("blue_white_green", ["blue", "white", "green"])

for column in reduced_df.iloc[:,:-2]:
    
    #print(f"Saving: {column}.svg")
    
    # Select the feature to color by
    color_feature = reduced_df.iloc[:,:-2][column]

    #columnClean = re.sub(r'[^\w\-_\. ]', '_', column)
    
    vmin = -0.4
    vmax = 0.4
    
    # Create a scatter plot colored by the selected feature
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        umap_df['UMAP1'], umap_df['UMAP2'], 
        c=color_feature, cmap=BWG, marker='o', alpha=0.8, 
        linewidths=0.09,vmin=vmin,vmax=vmax)
        
    #plt.colorbar(scatter, label=column)  # Add colorbar
    
    # Add a colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label(f"{column}", rotation=270, labelpad=15)
    
    # Customize colorbar ticks
    cbar.set_ticks([vmin,0,vmax])
    #cbar.ax.set_yticklabels(['Low (Blue)', 'Zero (White)', 'High (Red)'])
    
    # Set the minimum and maximum ticks, leaving the rest automatic
    #cbar.ax.set_ylim(min(umap_df[column]), max(umap_df[column]))  # Ensure the colorbar spans the full range
    #cbar.set_ticks([min(umap_df_df[column]), max(umap_df[column])])  # Set the ticks at min and max
    #cbar.update_ticks()

    # Remove the grid
    plt.grid(False)

    # axis limits
    #plt.xlim(-7.5, 20)  
    #plt.ylim(-4, 10)

    # Add labels and title
    #plt.savefig(f'D:/PhD Terzo anno/TESI/immagini/allZfyveRab5UMAPvisual{column}.svg',transparent=True)
    
    #plt.savefig(f'C:/Users/E14Ge/Desktop/Data_results/UMAP_features2_colControlRef2{column}.pdf')
    #plt.savefig(f'C:/Users/E14Ge/Desktop/Data_results/UMAP_features2_colControlRef2{column}.png',dpi=600)
    plt.show()
 
    
""" COMPARISON OF FEATURES BETWEEN CLUSTERS """

# VIOLIN PLOTS

# Get list of feature columns (excluding the cluster column)
features = [col for col in reduced_df.iloc[:,:-2].columns if col != 'Clusters']


custom_palette = ['purple', 'blue', 'green', 'yellow']

# Plot violin plots for each feature
for feature in features:
    plt.figure(figsize=(8, 6))
    ax = sns.violinplot(x='Clusters', y=feature, data=reduced_df, palette=custom_palette, linewidth=0.5)
    
    # Set transparency for each violin
    for patch in ax.collections:
        patch.set_alpha(0.5)  # Adjust transparency level (0 is fully transparent, 1 is opaque)
    
    plt.title(f'Violin Plot of {feature} Across Clusters')
    plt.xlabel('Cluster')
    plt.ylabel(feature)
    plt.tight_layout()
    
    plt.ylim(-1,1)
    
    #plt.savefig(f'D:/PhD Terzo anno/TESI/immagini/Rab11ViolinCNDTB{feature}.svg')
    #plt.savefig(f'C:/Users/E14Ge/Desktop/Data_results/UMAP_clusters_comparison/{feature}.pdf')
    #plt.savefig(f'C:/Users/E14Ge/Desktop/Data_results/UMAP_clusters_comparison/{feature}.png',dpi=600)
    
    plt.show()
    
    
""" STATISTICAL TEST  """


# I perform statistical test between the feature 
# distributions of the clusters. Results are presented in 
# matrix form and printed in an .xlsx format. In each sheet of
# the file are written the statistical results for each feature.
# For pairwise comparison I perform Mann-Whitney U tests.
# For multiple comparison I perform Mann-Whitney U tests with 
# Bonferroni correction

# I also print an heatmap where the rusults are visualized. Color shades 
# represent the fold change between the median of feature distribution per cell
# subpopulation. Instead, the circle size represents the -log10(P-value) 
# resulting from the statistical test

n_clusters = max(labels)


if n_clusters <= 2: 
    
    # Extract unique clusters
    clusters = reduced_df['Clusters'].unique()
   
    # Define the two clusters
    cluster1_data = reduced_df[reduced_df['Clusters'] == clusters[0]]
    cluster2_data = reduced_df[reduced_df['Clusters'] == clusters[1]]
    
    SM_final = pd.DataFrame()  # Empty DataFrame
    
    # Create an Excel writer
    with pd.ExcelWriter("statistical_results.xlsx") as writer:
        
        # Iterate over each feature in the dataframe (excluding 'Clusters')
        for Feat_select in reduced_df.iloc[:,:-1].columns:
            
            # Perform Mann-Whitney U test
            cluster1 = cluster1_data[Feat_select]
            cluster2 = cluster2_data[Feat_select]
            
            stat, p = mannwhitneyu(cluster1, cluster2, alternative='two-sided')
            fold_change = abs(np.median(cluster1)/np.median(cluster2))
            
            if p < 0.05:
                
                significance = 'True'
            else:
                
                significance = 'False'
            
            # Store results
            results = [{
                'Feature': Feat_select,
                'Cluster pairs':'1 vs 2',
                'U-statistic': stat,
                'p-value': p,
                'Fold_Change': fold_change,
                'Significant': significance
            }]
            
            
            # Convert to DataFrame and save to Excel
            statistical_matrix = pd.DataFrame(results)
            SM_final = pd.concat([SM_final, statistical_matrix], ignore_index=True) 
            statistical_matrix.to_excel(writer, sheet_name=Feat_select, index=False)
            
            
    # Apply transformation for circle size
    SM_final["Size"] = SM_final["p-value"].apply(lambda p: np.clip(-np.log10(p),0,20) if p <= 0.05 else np.nan)
    
    # Keep all features in the plot but drop NaN sizes from plotting circles
    df_filtered = SM_final.copy()
    
    # Define a custom colormap using RGB colors
    custom_colors = [
    (0.0, (0.6, 0.56, 0.76)),  # Violet
    (1.0, (0.95, 0.64, 0.25))  # Orange
    ]

    
    # Convert RGB to proper colormap format (0-1 scale)
    custom_cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", [x[1] for x in custom_colors])

    
    # Normalize fold-change median for color mapping
    norm = mcolors.Normalize(vmin=SM_final["Fold_Change"].min(), vmax=SM_final["Fold_Change"].max())
    
    # Convert expression values to RGB hex codes using the custom colormap
    df_filtered["Color"] = df_filtered["Fold_Change"].apply(lambda x: mcolors.rgb2hex(custom_cmap(norm(x))))
    
    # Create the dot plot
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot only significant features
    scatter = sns.scatterplot(
        data=df_filtered.dropna(subset=["Size"]),  # Drops only NaN sizes but keeps all features in the axis
        x="Cluster pairs", y="Feature",
        size="Size", sizes=(30, 800),  # Ensures visible range for significant features
        hue="Fold_Change", palette=custom_cmap,
        marker='o', linewidth=0.5
    )
    
    
    # Ensure all features appear on the y-axis, even if they have no circles
    plt.yticks(ticks=range(len(SM_final['Feature'])), labels=SM_final['Feature'])
    
    # Formatting    
    plt.xlabel("subpopulation pairwise")
    plt.ylabel("Features")
    plt.xticks(rotation=45)  # Rotate x-axis labels for better visibility
    plt.legend(title='Fold Change & p-value', bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
    plt.grid(False)
    plt.tight_layout()
    
    #plt.savefig('D:/PhD Terzo anno/TESI/immagini/Statistical_analysis.svg')
    
    # Show the plot
    plt.show()

else: 
    
    

    # Extract unique clusters
    clusters = reduced_df['Clusters'].unique()
    
    # Get all pairwise combinations of clusters
    cluster_pairs = list(combinations(clusters, 2))
    
    SM_final = pd.DataFrame()  # Empty DataFrame
    
    # Create an Excel writer
    with pd.ExcelWriter("statistical_results.xlsx") as writer:
        
        # Iterate over each feature in the dataframe (excluding 'Clusters')
        for Feat_select in reduced_df.iloc[:,:-3].columns:
    
            # Perform Mann-Whitney U tests for each pair
            p_values = []
            results = []
            for pair in cluster_pairs:
                cluster1 = reduced_df[reduced_df['Clusters'] == pair[0]][Feat_select]
                cluster2 = reduced_df[reduced_df['Clusters'] == pair[1]][Feat_select]
                
                stat, p = mannwhitneyu(cluster1, cluster2, alternative='two-sided')
                p_values.append(p)
                fold_change = abs(np.median(cluster1)/np.median(cluster2))
                results.append(
                    {'Feature': Feat_select,
                     'Cluster pairs': f'{pair[0]} vs {pair[1]}', 
                     'U-statistic': stat, 
                     'p-value': p, 
                     'Fold_Change': fold_change
                     
                     })
            
            # Apply Bonferroni correction
            _, corrected_p_values, _, _ = multipletests(p_values, alpha=0.05, method='bonferroni')
            
            # Add corrected p-values to results
            for i, result in enumerate(results):
                result['Corrected p-value'] = corrected_p_values[i]
                result['Significant'] = corrected_p_values[i] < 0.05  # Whether the test is significant
            
            # Convert to DataFrame and save to Excel
            statistical_matrix = pd.DataFrame(results)
            SM_final = pd.concat([SM_final, statistical_matrix], ignore_index=True)
            statistical_matrix.to_excel(writer, sheet_name=Feat_select, index=False)
 
       


    # Define a custom colormap using RGB colors
    custom_colors = [
    (0.0, (0.6, 0.56, 0.76)),  # Violet
    (1.0, (0.95, 0.64, 0.25))  # Orange
    ]
    
    custom_cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", custom_colors, N=256)
    
    # Compute -log10(Corrected p-value)
    SM_final["-log10(Corrected p-value)"] = -np.log10(SM_final["Corrected p-value"])
    
    # Filter only significant results
    df_significant = SM_final[SM_final["Significant"]]
    
    # Define colorbar range
    vmin = 0
    vmax = df_significant["Fold_Change"].max()
    
    # Define size legend values
    size_legend_values = [1.5, 2.5, 5, 7.5,10]  # Example significance levels
    size_legend_sizes = [val * 100 for val in size_legend_values]  # Scale for visibility
    
    # Create figure and axis
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(
        x=df_significant["Cluster pairs"],
        y=df_significant["Feature"],
        s=df_significant["-log10(Corrected p-value)"] * 10,  # Scale size for visibility
        c=df_significant["Fold_Change"],
        cmap=custom_cmap,  # Apply custom colormap
        alpha=0.75,
        edgecolors="black",
        vmin=vmin,
        vmax=vmax
    )
    
    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label("Fold Change")
    cbar.set_ticks([vmin, vmax])
    cbar.set_ticklabels([f"{vmin:.2f}", f"{vmax:.2f}"])
    
    # Labels and title
    plt.xlabel("Cluster Pairs")
    plt.ylabel("Feature")
    plt.title("Dot Plot of Features vs Cluster Pairs")
    
    # Add size legend
    for size, value in zip(size_legend_sizes, size_legend_values):
        plt.scatter([], [], s=size, color='gray', alpha=0.5, label=f"-log10(p) = {value}")
    
    plt.legend(title="Circle Size", loc="upper right", bbox_to_anchor=(1.3, 1))
    
    # Improve layout
    plt.xticks(rotation=45, ha="right")
    plt.grid(False)
    plt.tight_layout()
    
    #plt.savefig('D:/PhD Terzo anno/TESI/immagini/StatisticAllRab11.svg')
    #plt.savefig('D:/PhD Terzo anno/TESI/immagini/Statistical_analysis.svg')
    #plt.savefig('D:/PhD Terzo anno/TESI/immagini/Statistical_analysis.pdf')
    
    # Show plot
    plt.show() 
    
    
    
""" CELL PROPORTION (KS>0/KS<0) PER FEATURE AND PER SUBPOPULATION """
   
# Define colors for positive and negative fractions
colors = ['#2ca02c', '#1f77b4']  # green for positive, blue for negative 
   
  
# iterate on the number of subpopulation identified
for cluster in reduced_df['Clusters'].unique():
    
    # Filter only datapoints belonging to subpopulation n
    df_cluster = reduced_df[reduced_df['Clusters'] == cluster].copy()
    
    # Compute fraction of positive and negative datapoints per feature
    feature_fractions = pd.DataFrame({
        'KS > 0': (df_cluster[reduced_df.iloc[:, :-3].columns] > 0).mean(),
        'KS < 0': (df_cluster[reduced_df.iloc[:, :-3].columns] < 0).mean()
    })
    
    # Create the stacked bar plot
    fig, ax = plt.subplots(figsize=(8, 6))
    bottom = np.zeros(len(reduced_df.iloc[:, :-3].columns))
    
   
    for idx, category in enumerate(['KS > 0', 'KS < 0']):
        ax.bar(reduced_df.iloc[:, :-3].columns, feature_fractions[category], bottom=bottom, color=colors[idx], label=category)
        bottom += feature_fractions[category]  # Stack bars
    
    # Formatting
    ax.set_xlabel('Features')
    ax.set_ylabel('Proportion')
    ax.set_yticks(np.arange(0, 1.1, 0.25))
    plt.xticks(rotation=45)
    ax.legend(title="Sign")
    
    #plt.savefig(f'D:/PhD Terzo anno/TESI/immagini/KSproportionRab11_CDNTN{cluster}.svg')
    
    # Show the plot
    plt.show()
                
    

""" FEATURE CLASSIFICATION """


# I run Random Forest algorithm to get feature importance
# and to quantify which feature best distinguish one cluster
# from the others.

 
# Remove eventual outliers (label -1 from DBSCAN) =
filtered_df = reduced_df[reduced_df.iloc[:, -1] != -1]

# Separate features and cluster labels 
X = filtered_df.iloc[:, :-2]
y = filtered_df.iloc[:, -2]
unique_clusters = np.sort(y.unique())
n_clusters = len(unique_clusters)

# Compute feature importances 
feature_importances = []
features_name = X.columns.tolist()

if n_clusters <= 2: # in this case I aply binary classification
    
    rf = RandomForestClassifier(random_state=42, n_estimators=100)
    rf.fit(X, y)
    feature_importances.append(rf.feature_importances_)
    feature_importances.append(rf.feature_importances_)

    # Use actual cluster labels as row label
    heatmap_index = ["1","2"]

# if I have more that 2 clusters, I want to measure the feature importance in 
# classifying cells belonging to a clusters with repsect to the others
else:
    
    lb = LabelBinarizer()
    y_bin = lb.fit_transform(y)

    # Ensure y_bin is 2D
    if y_bin.ndim == 1:
        y_bin = y_bin.reshape(-1, 1)

    for i, cluster in enumerate(lb.classes_):
        y_binary = y_bin[:, i]  # current cluster vs rest

        rf = RandomForestClassifier(random_state=42, n_estimators=100)
        rf.fit(X, y_binary)
        feature_importances.append(rf.feature_importances_)

    # Use actual cluster labels as row labels
    heatmap_index = [str(cl) for cl in lb.classes_]



# Plotting

# For sake of simplicity I store all the information in a dataframe
importance_df = pd.DataFrame(
    feature_importances,
    index=heatmap_index,
    columns=features_name
)

plt.figure(figsize=(10, 0.8 * len(importance_df)))  # auto-height
sns.heatmap(
    importance_df,
    cmap="Greys",
    annot=False,
    cbar=True,
    vmax=1,
    vmin=0,
    linecolor="grey",
    linewidths=0.5
)
plt.title("Feature Importance Heatmap")
plt.ylabel("Cluster")
plt.xlabel("Feature")
plt.tight_layout()

#plt.savefig('D:/PhD Terzo anno/TESI/immagini/Feature importanceRab11CTRL.svg')
#plt.savefig('C:/Users/E14Ge/Desktop/Data_results/UMAP_clusters_comparison/Feat_importanceClusterHM.pdf')
#plt.savefig('C:/Users/E14Ge/Desktop/Data_results/UMAP_clusters_comparison/Feat_importanceClusterHM.png',dpi=600)

plt.show()