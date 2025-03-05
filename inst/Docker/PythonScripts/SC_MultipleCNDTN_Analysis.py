import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import MyFunctions2 as F 
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import seaborn as sns
import umap
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelBinarizer
from itertools import combinations
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

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
Data_path = 'C:/Users/E14Ge/Desktop/NewData il mio paper/dati singole cellule e farmaci KS/rab11/ALL.xlsx'

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

#plt.savefig('C:/Users/E14Ge/Desktop/Data_results/allTRTDRab5UMAP.svg',transparent=True)
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

#plt.savefig('C:/Users/E14Ge/Desktop/Data_results/allTRTDRab5UMAP.svg',transparent=True)
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

# plot
plt.figure(figsize=(8, 6))
for treatment in umap_df["Treatment"].unique():
    subset = umap_df[umap_df["Treatment"] == treatment]
    plt.scatter(
        subset["UMAP1"], subset["UMAP2"], label=f"{treatment}", s = 8, alpha=0.5
    )
plt.title("UMAP Visualization by Treatment")
plt.xlabel("UMAP1")
plt.ylabel("UMAP2")
plt.legend()
plt.grid(False)
#plt.savefig('C:/Users/E14Ge/Desktop/Data_results/allTRTDRab5UMAP.svg',transparent=True)
#plt.savefig('C:/Users/E14Ge/Desktop/Data_results/allTRTDRab5UMAP.pdf')
#plt.savefig('C:/Users/E14Ge/Desktop/Data_results/allTRTDRab5UMAP.png',dpi=600)
plt.show()


""" CLUSTERING """

# Here I perform clustering by KMeans. To unbiasly decide
# the number of cluster I get help from elbow method and 
# Silhouette Score


### ELBOW METHOD ###

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
# based on Elbow and Silhouette score

# Define number of clusters
n_clusters = optimal_k

# Initialize k-means
kmeans = KMeans(n_clusters=n_clusters, random_state=42)

# Fit k-means to your data
kmeans.fit(Data_UMAP)

# Get cluster labels
labels = kmeans.labels_


""" UMAP PLOT WITH CLUSTERS """
plt.figure(figsize=(8, 6))

# Create a colormap for clusters
colors = plt.cm.viridis(np.linspace(0, 1, n_clusters))

for cluster_id in range(n_clusters):
    cluster_points = umap_df[labels == cluster_id]
    plt.scatter(
        cluster_points['UMAP1'], cluster_points['UMAP2'],
        c=[colors[cluster_id]],  # Assign cluster-specific color
        label=f"Cluster {cluster_id + 1}",
        s=50, alpha=0.7
    )


# Add legend
plt.legend(title="Clusters")
plt.title('K-Means Clustering on UMAP Data')
plt.xlabel('UMAP Dimension 1')
plt.ylabel('UMAP Dimension 2')

#plt.savefig('C:/Users/E14Ge/Desktop/Data_results/ControlRab11UMAP_ref2.svg',transparent=True)
#plt.savefig('C:/Users/E14Ge/Desktop/Data_results/ControlRab11UMAP_ref2.pdf')
#plt.savefig('C:/Users/E14Ge/Desktop/Data_results/ControlRab11UMAP_ref2.png',dpi=600)

plt.show()



""" FEATURES VALUES VISUALIZATION IN UMAP """

# I visualize in the UMAP space the features values. I use
# only the features selected after feature selection

for column in reduced_df:
    
    #print(f"Saving: {column}.svg")
    
    # Select the feature to color by
    color_feature = reduced_df[column]

    #columnClean = re.sub(r'[^\w\-_\. ]', '_', column)
    
    vmin = -0.5
    vmax = 0.5
    
    # Create a scatter plot colored by the selected feature
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        umap_df['UMAP1'], umap_df['UMAP2'], 
        c=color_feature, cmap='bwr', marker='o', alpha=0.8, 
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

    # Add axes crossing at the origin
    plt.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)  # Horizontal line at y=0
    plt.axvline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)  # Vertical line at x=0

    # Remove the grid
    plt.grid(False)

    # axis limits
    #plt.xlim(-7.5, 20)  
    #plt.ylim(-4, 10)

    # Add labels and title
    #plt.savefig(f'C:/Users/E14Ge/Desktop/Data_results/UMAP_features2_colControlRef2{column}.svg')
    #plt.savefig(f'C:/Users/E14Ge/Desktop/Data_results/UMAP_features2_colControlRef2{column}.pdf')
    #plt.savefig(f'C:/Users/E14Ge/Desktop/Data_results/UMAP_features2_colControlRef2{column}.png',dpi=600)
    plt.show()
 
    
""" 
Le rimanenti parti di caratterizzazione delle singole sottopopoplazioni le devo
ancora un attimo pensare bene.

"""
