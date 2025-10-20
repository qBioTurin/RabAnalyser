import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import MyFunctions2 as F 
import seaborn as sns
import umap
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelBinarizer
from itertools import combinations
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

from sklearn.model_selection import train_test_split

import igraph as ig
import leidenalg
from sklearn.metrics import silhouette_score, davies_bouldin_score

import matplotlib.patches as mpatches

import matplotlib.colors as mcolors ###
import matplotlib.cm as cm ###
from statsmodels.stats.proportion import proportions_ztest ###

from scipy.stats import chi2_contingency

import os


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
# Specify the folder containing the .xlsx files
Data_path = 'D:/Ricerca/Progetti/RabAnalyser/feature_extraction/SC_analysis_validation_V2.xlsx'

# Load the Excel file into a DataFrame
df = pd.read_excel(Data_path)

# Define noise floor
qalpha = 0.10  
gamma = 0.05

# Separate label column (assume it's the first column)
labels = df.iloc[:, -1]        # condition labels
df_num = df.iloc[:, :-1]       # all numeric columns

# Apply soft-threshold to numeric data
df_num = df_num.apply( lambda col: np.sign(col) * (np.abs(col) - qalpha) * np.tanh((np.abs(col) - qalpha/gamma)) )  #df_num.apply(lambda col: np.sign(col) * np.maximum(np.abs(col) - qalpha, 0))

# Recombine
df = pd.concat([df_num, labels], axis=1)
    
# Get the directory where to save the plots
output_dir = os.path.dirname(Data_path)

""" SUBSAMPLING  """

# Step 1: Find minimum count across all treatments
min_count = df['Class'].value_counts().min()

# Step 2: Subsample each treatment to min_count rows
df = (
    df
    .groupby('Class', group_keys=False)
    .apply(lambda x: x.sample(n=min_count, random_state=42))  # random_state ensures reproducibility
    .reset_index(drop=True)
)

print(df['Class'].value_counts())


# Remove the column with treatment labels
# If there is a column with a condition to be related remove also that

df_selected = df.iloc[:, :-1]



""" FEATURE SELECTION """
print("data filtering")

#here it is performed the feature lection step

# values pearson coefficient to be used as threshold
threshold = 0.7
reduced_df = F.FilterFeat(df_selected, threshold)


### CORRELATION MATRIX PLOT ###

# compute the correlation matrix of the original features
correlation_matrixOLD = df_selected.corr()


output_pathPNG = os.path.join(output_dir, "CorrMatrix_old.png")
output_pathSVG = os.path.join(output_dir, "CorrMatrix_old.svg")

# Plot the correlation matrix of the original features
plt.figure(figsize=(12, 10))
sns.heatmap(
    correlation_matrixOLD,
    annot=True,      # Set to True if you want to show correlation values
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

#plt.savefig(output_pathSVG,transparent=True)
#plt.savefig(output_pathPNG,dpi=600)

plt.show()


# compute the correlation matrix of the original features
correlation_matrixNEW = reduced_df.corr()

output_pathPNG = os.path.join(output_dir, "CorrMatrix_new.png")
output_pathSVG = os.path.join(output_dir, "CorrMatrix_new.svg")

# Plot the correlation matrix of the filtered features
plt.figure(figsize=(12, 10))
sns.heatmap(
    correlation_matrixNEW,
    annot=True,      # Set to True if you want to show correlation values
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

#plt.savefig(output_pathSVG,transparent=True)
#plt.savefig(output_pathPNG,dpi=600)

plt.show()


""" UMAP ANALYSIS """


# Standardize the data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(reduced_df)


# Initialize UMAP
reducer = umap.UMAP(n_neighbors=20, min_dist=1, n_components=2, metric='correlation', random_state=42) #metric='cosine'

# Fit and transform the data
Data_UMAP = reducer.fit_transform(data_scaled)

#create a dataframe with the umap dimension as columns
umap_df = pd.DataFrame(Data_UMAP, columns=["UMAP1", "UMAP2"])


""" UMAP PLOT """

# extract conditions labels 
TRTD_label = df['Class']

# add conditions labels to UMAP dataframe
umap_df = pd.DataFrame(Data_UMAP, columns=["UMAP1", "UMAP2"])
umap_df["Class"] = TRTD_label


# === Toggle: show_all = True to plot all conditions, False to show only one ===
show_all = True  # Set to False to show only one condition
highlight_treatment = 'Ref'  # Only used if show_all is False

# Prepare figure
plt.figure(figsize=(8, 6))

if show_all:
    unique_treatments = umap_df["Class"].unique()
    
    # Create a colormap and normalize it to number of treatments
    cmap = cm.get_cmap('tab10', len(unique_treatments))  # 'tab20' for more colors
    norm = mcolors.Normalize(vmin=0, vmax=len(unique_treatments)-1)
    
    # Plot each treatment with a different color
    for i, cond in enumerate(unique_treatments):
        subset = umap_df[umap_df["Class"] == cond]
        color = cmap(norm(i))
        plt.scatter(subset["UMAP1"], subset["UMAP2"], label=cond, s=30, alpha=0.7, c=[color])
else:
    # Only one treatment, using a fixed color
    subset = umap_df[umap_df["Class"] == highlight_treatment]
    plt.scatter(subset["UMAP1"], subset["UMAP2"], label=highlight_treatment, s=30, alpha=0.7, c=(0.6, 0.6, 0.6))

# Plot styling
plt.title("UMAP Visualization by Treatment")
plt.xlabel("UMAP1")
plt.ylabel("UMAP2")
plt.legend()
plt.grid(False)
plt.tight_layout()

plt.show()




""" CLUSTERING """

# Here I perform clustering by Leiden alorithm. Morevoer, to unbiasly decide
# the number of clusters I get help from several scores/indeces such as:
# Silhouette Score, elbow method, DBI score and modularity score. 


### LEIDEN ###

output_pathPNG = os.path.join(output_dir, "UMAP_clusters.png")
output_pathSVG = os.path.join(output_dir, "UMAP_clusters.svg")

# for the Leiden algorithm the number of clusters depends on the resolution
# parameter. Here, I test several resolution parameters 
# Define a range of resolution values for Leiden clustering
resolution_values = np.linspace(0.0001, 0.1, 50)  # Testing different granularity levels

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
        silhouette_scores.append(silhouette_score(data_scaled, labels))
        dbi_scores.append(davies_bouldin_score(data_scaled, labels))
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

#plt.savefig('C:/Users/E14Ge/Desktop/Rab cluster data/single cell analysis/esp_GLIOMAcells/GLIOMAcellsRab11_LeidenScores.png',transparent=True)

plt.title("Silhouette & DBI Scores")

# Plot Metrics vs. Number of Clusters
plt.figure(figsize=(10, 5))
plt.plot(resolution_values, modularity_scores, marker="o", color="red", markersize=3, label="Modularity Score")
plt.xlabel("resolutions values (a.u.)")
plt.ylabel("Modularity Score")
plt.legend()
#plt.title("Modularity vs. Cluster Number")

#plt.savefig(output_pathSVG,transparent=True)
#plt.savefig(output_pathPNG,dpi=600)

plt.show()


#choose the more appropriate resolution parameter
res = 0.004

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
''

plt.figure(figsize=(8, 6))
scatter = plt.scatter(umap_df["UMAP1"], umap_df["UMAP2"], c=umap_df["Leiden_Cluster"], cmap="viridis", alpha=0.7, s=15)

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

#plt.savefig(output_pathSVG,transparent=True)
#plt.savefig(output_pathPNG,dpi=600)

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

for idx, value in enumerate(unique_values):
    ax.bar("Proportion", proportions[idx], bottom=bottom, color=colors[idx], label=f"{value}")
    bottom += proportions[idx]  # Update bottom for stacking

# Formatting
ax.set_ylabel("Cell proportion")
ax.set_yticks(np.arange(0, 1.1, 0.25))
ax.legend(title="Subpopulations", bbox_to_anchor=(1.05, 1), loc='upper left')

#plt.savefig('D:/PhD Terzo anno/TESI/immagini/allRab11proportion_subpop.svg',transparent=True)

# Show the plot
plt.show()


""" CELL PROPORTION IN CONTROL AND TREATED CONDITION """

# I quantify the proportion of cells for each condition and in each subpopulation.
# Proportions are calculate with respect to the total number of cells belonging
# to each condition

# add information about conditions of the cells: control or treatment
reduced_df['Condition'] = df['Class']

output_pathPNG = os.path.join(output_dir, "Cell_Prop_sub.png")
output_pathSVG = os.path.join(output_dir, "Cell_Prop_sub.svg")

# Get all unique conditions
conditions = reduced_df['Condition'].unique()

# Count total number of cells per condition
total_cells_per_condition = reduced_df['Condition'].value_counts().to_dict()

# Count cells per cluster per condition
cluster_counts = reduced_df.groupby(["Clusters", "Condition"]).size().unstack(fill_value=0)

# Normalize counts by total cells in each condition
cluster_fractions = cluster_counts.copy()
for condition in conditions:
    cluster_fractions[condition] = (cluster_fractions[condition] / total_cells_per_condition[condition])*100

# Set up plot
clusters = cluster_fractions.index
x = np.arange(len(clusters))  # positions for each cluster
bar_width = 0.8 / len(conditions)  # width of each bar (fit all bars within total width = 0.8)

fig, ax = plt.subplots(figsize=(10, 6))

# Use color palette
colors = plt.cm.tab20.colors

# Plot each condition
for i, condition in enumerate(conditions):
    offset = (i - len(conditions)/2) * bar_width + bar_width/2
    ax.bar(x + offset, cluster_fractions[condition], width=bar_width,
           label=condition, color=colors[i % len(colors)])

# Formatting
ax.set_ylim([0, 110])
ax.set_xticks(x)
ax.set_xticklabels(clusters, rotation=45)
ax.set_xlabel("Subpopulations")
ax.set_ylabel("Cell proportion")
ax.set_title("Comparison of Cell Fractions Across Conditions Per Subpopulation")
ax.legend(title="Condition")


#plt.savefig(output_pathSVG,transparent=True)
#plt.savefig(output_pathPNG,dpi=600)

plt.tight_layout()
plt.show()



# I quantify the statistical significance among the differences of the various 
# proportion by Z-test. I adjusted the p-value by Bonferroni correction. Results
# are returned as a table.

# STEP 1: Prepare contingency table (clusters x conditions)
contingency = reduced_df.groupby(['Clusters', 'Condition']).size().unstack(fill_value=0)

# STEP 2: Global Chi-square test
chi2_stat, p_global, dof, expected = chi2_contingency(contingency)
print(f"\nGlobal Chi-square test p-value = {p_global:.4e}")

# Store pairwise results
results = []

# STEP 3: If global test is significant, do post hoc comparisons
if p_global < 0.05:
    print("Global test is significant. Proceeding with pairwise post hoc tests...\n")

    raw_p_values = []
    test_records = []

    for cluster in contingency.index:
        for cond1, cond2 in combinations(contingency.columns, 2):
            count = np.array([
                contingency.loc[cluster, cond1],
                contingency.loc[cluster, cond2]
            ])
            nobs = np.array([
                contingency[cond1].sum(),
                contingency[cond2].sum()
            ])

            # Z-test or Fisher's exact test
            if np.all(count >= 5):
                stat, p = proportions_ztest(count, nobs)
            else:
                from scipy.stats import fisher_exact
                table = [[count[0], nobs[0] - count[0]],
                         [count[1], nobs[1] - count[1]]]
                _, p = fisher_exact(table)

            raw_p_values.append(p)
            test_records.append((cluster, cond1, cond2, p))

    # Adjust p-values
    adjusted_p_values = multipletests(raw_p_values, method='fdr_bh')[1]

    # Build and print result table
    print("Pairwise Post Hoc Test Results:\n")
    print(f"{'Cluster':<15} {'Condition 1':<15} {'Condition 2':<15} {'Raw p-value':<15} {'Adjusted p':<15} {'Significant'}")
    print("-" * 85)

    for (cluster, cond1, cond2, raw_p), adj_p in zip(test_records, adjusted_p_values):
        significant = "Yes" if adj_p < 0.05 else "No"
        print(f"{cluster:<15} {cond1:<15} {cond2:<15} {raw_p:<15.4e} {adj_p:<15.4e} {significant}")

else:
    print("Global test not significant. No pairwise comparisons performed.")





""" FEATURES VALUES VISUALIZATION IN UMAP """

# I visualize in the UMAP space the features values. I use
# only the features selected after feature selection


# Define a custom colormap transitioning from Blue to White to Green
BWG = mcolors.LinearSegmentedColormap.from_list("blue_white_green", ["blue", "white", "green"])

for column in reduced_df.iloc[:,:-2]:
    
    #print(f"Saving: {column}.svg")
    
    output_pathPNG = os.path.join(output_dir, f'UMAP_visual{column}.png')
    output_pathSVG = os.path.join(output_dir, f'UMAP_visual{column}.svg')
    
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
    #plt.savefig(output_pathSVG,transparent=True)   
    #plt.savefig(output_pathPNG,dpi=600)
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

n_clusters = max(labels)+1 # 0 is a label

output_pathPNG = os.path.join(output_dir, "Statistics.png")
output_pathSVG = os.path.join(output_dir, "Statistics.svg")


if n_clusters <= 2: 
    
    # Extract unique clusters
    clusters = reduced_df['Clusters'].unique()
   
    # Define the two clusters
    cluster1_data = reduced_df[reduced_df['Clusters'] == clusters[0]]
    cluster2_data = reduced_df[reduced_df['Clusters'] == clusters[1]]
    
    SM_final = pd.DataFrame()  # Empty DataFrame
    
    # Create an Excel writer
    with pd.ExcelWriter("statistical_results.xlsx") as writer:
        
        # Iterate over each feature in the dataframe (excluding 'Clusters' and 'conditions')
        for Feat_select in reduced_df.iloc[:,:-2].columns:
            
            # Perform Mann-Whitney U test
            cluster1 = cluster1_data[Feat_select]
            cluster2 = cluster2_data[Feat_select]
            
            stat, p = mannwhitneyu(cluster1, cluster2, alternative='two-sided')
            fold_change = np.median(cluster1) - np.median(cluster2)
            
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
    
    BWG = mcolors.LinearSegmentedColormap.from_list("blue_white_green", ["blue", "white", "green"])

    
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
        hue="Median differece", palette=BWG,
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
    
    plt.savefig(output_pathSVG,transparent=True)
    plt.savefig(output_pathPNG,dpi=600)
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
        for Feat_select in reduced_df.iloc[:,:-2].columns:
    
            # Perform Mann-Whitney U tests for each pair
            p_values = []
            results = []
            for pair in cluster_pairs:
                cluster1 = reduced_df[reduced_df['Clusters'] == pair[0]][Feat_select]
                cluster2 = reduced_df[reduced_df['Clusters'] == pair[1]][Feat_select]
                
                stat, p = mannwhitneyu(cluster1, cluster2, alternative='two-sided')
                p_values.append(p)
                #fold_change = abs(np.median(cluster1)/np.median(cluster2))
                fold_change = np.median(cluster1) - np.median(cluster2)
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
    
    BWG = mcolors.LinearSegmentedColormap.from_list("blue_white_green", ["blue", "white", "green"])
    
    # Compute -log10(Corrected p-value)
    SM_final["-log10(Corrected p-value)"] = -np.log10(SM_final["Corrected p-value"])
    
    # Filter only significant results
    df_significant = SM_final[SM_final["Significant"]]
    
    # Define colorbar range
    vmin = df_significant["Fold_Change"].min()
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
        cmap=BWG,  # Apply custom colormap
        alpha=0.75,
        edgecolors="black",
        vmin=vmin,
        vmax=vmax
    )
    
    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label("Median difference")
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
    
    plt.savefig(output_pathSVG,transparent=True)
    plt.savefig(output_pathPNG,dpi=600)
    
    # Show plot
    plt.show() 
    
    
    
""" FINGERPRINTS FOR EACH SUBPOPULATION """
            
"""
KS < 0.15 -> small effect
KS 0.15-0.25 -> moderate effect
KS > 0.3 -> strong effect

Arbitrary threshold given by visual inspection of distributions shift

"""
# empty dataframe containing the mean KS per feature and per cluster
df_finger = pd.DataFrame(columns=reduced_df.iloc[:, :-2].columns)

# iterate on the number of subpopulation identified
for cluster in reduced_df['Clusters'].unique():
    
    # Filter only datapoints belonging to subpopulation n
    df_cluster = reduced_df[reduced_df['Clusters'] == cluster].copy()
    
    # average the features values
    df_mean = df_cluster.iloc[:, :-2].mean(axis=0).to_frame().T
    
    df_finger = pd.concat([df_finger, df_mean], axis=0, ignore_index=True)

# add cluster names as indeces of the dataframe and I sort the indeces   
df_finger.index = reduced_df['Clusters'].unique()   
df_finger = df_finger.sort_index()   


# heatmap plot

# Blue gradient for left side (dark → light)
left_colors  = ["#2166ac", "#67a9cf"]   # dark blue → light blue

# Green gradient for right side (light → dark)
right_colors = ["#b2e2b2", "#1b7837"]   # light green → dark green


vmin, vmax = -0.5, 0.5
white_min, white_max = -0, 0

cmap = F.make_whiteband_cmap(
    left_colors=left_colors,
    right_colors=right_colors,
    vmin=vmin, vmax=vmax,
    white_min=white_min, white_max=white_max,
    white_color="white",   # or (1,1,1,1)
    n=512
)

plt.figure(figsize=(10, 6))
sns.heatmap(
    df_finger,
    cmap=cmap,
    vmin=vmin, vmax=vmax,  
    cbar=True
)
plt.title(f"Custom heatmap with white band [{white_min}, {white_max}]")
plt.xlabel("Features")
plt.ylabel("Clusters")
plt.show()



""" FEATURE CLASSIFICATION """

# I run Random Forest algorithm to get feature importance
# and to quantify which feature best distinguish one cluster
# from the others.

output_pathPNG = os.path.join(output_dir, "feat_classification.png")
output_pathSVG = os.path.join(output_dir, "feat_classification.svg")

 
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
    annot=True,
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

#plt.savefig(output_pathSVG,transparent=True)
#plt.savefig(output_pathPNG,dpi=600)


plt.show()


