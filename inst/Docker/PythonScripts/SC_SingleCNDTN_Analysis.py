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
identification and characterization. Here, each single 
condition is considered singularly.

input: KS matrix

output:
    
Feature correlation matrices
UMAP plot
UMAP plot with clusters
Feature values visualization in UMAP
Test statistici
Radar plot
Violin plot
Feature importance plot
 

"""


""" File loading """
print("file loading")
Data_path = sys.argv[1] # 'Data_path = 'C:/Users/E14Ge/Desktop/NewData il mio paper/dati singole cellule e farmaci KS/rab11/Nocodazole_SCrab11_withControlRef2.xlsx'

# Load the Excel file into a DataFrame
df = pd.read_excel(Data_path)


""" FEATURE SELECTION """
print("data filtering")

#here it is performed the feature lection step

# values pearson coefficient to be used as threshold
threshold = 0.75 
reduced_df = F.FilterFeat(df, threshold)


### CORRELATION MATRIX PLOT ###

# compute the correlation matrix of the original features
correlation_matrixOLD = df.corr()

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

plt.figure(figsize=(8, 6))

plt.scatter(umap_df['UMAP1'], umap_df['UMAP2'], s=30, alpha=0.8)

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



""" COMPARISON OF FEATURES BETWEEN CLUSTERS """



# VIOLIN PLOTS
reduced_df['Clusters'] = labels

# Get list of feature columns (excluding the cluster column)
features = [col for col in reduced_df.columns if col != 'Clusters']


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
    #plt.savefig(f'C:/Users/E14Ge/Desktop/Data_results/UMAP_clusters_comparison/{feature}.svg')
    #plt.savefig(f'C:/Users/E14Ge/Desktop/Data_results/UMAP_clusters_comparison/{feature}.pdf')
    #plt.savefig(f'C:/Users/E14Ge/Desktop/Data_results/UMAP_clusters_comparison/{feature}.png',dpi=600)
    plt.show()
    
    
    
""" STATISTICAL TEST  """


# I perform statistical test between the feature 
# distributions of the clusters. Results are presented in 
# matrix form and printed in an .xlsx format. In each sheet of
# the file are wrote the statistical results for each feature.
# For pairwise comparison I perform Mann-Whitney U tests.
# For multiple comparison I perform Mann-Whitney U tests with 
# Bonferroni correction


if n_clusters <= 2: 
    
    # Extract unique clusters
    clusters = reduced_df['Clusters'].unique()
   
    # Define the two clusters
    cluster1_data = reduced_df[reduced_df['Clusters'] == clusters[0]]
    cluster2_data = reduced_df[reduced_df['Clusters'] == clusters[1]]
    
    # Create an Excel writer
    with pd.ExcelWriter("statistical_results.xlsx") as writer:
        
        # Iterate over each feature in the dataframe (excluding 'Clusters')
        for Feat_select in reduced_df.iloc[:,:-1].columns:
            
            # Perform Mann-Whitney U test
            cluster1 = cluster1_data[Feat_select]
            cluster2 = cluster2_data[Feat_select]
            
            stat, p = mannwhitneyu(cluster1, cluster2, alternative='two-sided')
            median_diff = np.median(cluster1) - np.median(cluster2)
            
            if p < 0.05:
                
                significance = 'True'
            else:
                
                significance = 'False'
            
            # Store results
            results = [{
                'Feature': Feat_select,
                'U-statistic': stat,
                'p-value': p,
                'Median_Diff': median_diff,
                'Significant': significance
            }]
            
            
            # Convert to DataFrame and save to Excel
            statistical_matrix = pd.DataFrame(results)
            statistical_matrix.to_excel(writer, sheet_name=Feat_select, index=False)

else: 
    
    

    # Extract unique clusters
    clusters = reduced_df['Clusters'].unique()
    
    # Get all pairwise combinations of clusters
    cluster_pairs = list(combinations(clusters, 2))
    
    # Create an Excel writer
    with pd.ExcelWriter("statistical_results.xlsx") as writer:
        
        # Iterate over each feature in the dataframe (excluding 'Clusters')
        for Feat_select in reduced_df.iloc[:,:-1].columns:
    
            # Perform Mann-Whitney U tests for each pair
            p_values = []
            results = []
            for pair in cluster_pairs:
                cluster1 = reduced_df[reduced_df['Clusters'] == pair[0]][Feat_select]
                cluster2 = reduced_df[reduced_df['Clusters'] == pair[1]][Feat_select]
                
                stat, p = mannwhitneyu(cluster1, cluster2, alternative='two-sided')
                p_values.append(p)
                median_diff = np.median(cluster1) - np.median(cluster2)
                results.append({'Clusters pair': f'{pair[0]} vs {pair[1]}', 'U-statistic': stat, 'p-value': p, 'Median_Diff': median_diff})
            
            # Apply Bonferroni correction
            _, corrected_p_values, _, _ = multipletests(p_values, alpha=0.05, method='bonferroni')
            
            # Add corrected p-values to results
            for i, result in enumerate(results):
                result['Corrected p-value'] = corrected_p_values[i]
                result['Significant'] = corrected_p_values[i] < 0.05  # Whether the test is significant
            
            # Convert to DataFrame and save to Excel
            statistical_matrix = pd.DataFrame(results)
            statistical_matrix.to_excel(writer, sheet_name=Feat_select, index=False)


""" SPIDER PLOT """

# Perform the spider plot with the median features values
# for each cluster

# Extract unique clusters
clusters = reduced_df['Clusters'].unique()

# Group by the 'Cluster' column and calculate the median for each feature
df_median_per_cluster = reduced_df.groupby('Clusters').median()

# Dataframe where the cluster labels are stored as indeces
Median_df = pd.DataFrame(df_median_per_cluster, index=clusters)

# Sort the DataFrame by its index
Median_df = Median_df.sort_index()

# Compute angles for the radar chart
num_features = len(Median_df.columns)
angles = np.linspace(0, 2 * np.pi, num_features, endpoint=False).tolist()
angles += angles[:1]  # Close the loop

# Adjust the y-axis range to include negative values
min_value = Median_df.min().min()  # Minimum value in the DataFrame
max_value = Median_df.max().max()  # Maximum value in the DataFrame

# Initialize the plot for radial bars
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

# Plot each cluster using radial bars
bar_width = 2 * np.pi / num_features / len(Median_df)  # Adjust bar width
colors = plt.cm.viridis(np.linspace(0, 1, len(Median_df)))  # Generate colors


for idx, (cluster_name, row) in enumerate(Median_df.iterrows()):
    values = row.tolist() + [row.iloc[0]]  # Close the loop
    for i, value in enumerate(values[:-1]):
        angle = angles[i] + idx * bar_width + bar_width/2
        #angle = angles[i] - bar_width / 2 + idx * bar_width  # Offset bars for each cluster
        ax.bar(
            angle,
            value,
            width=bar_width,
            color=colors[idx],
            edgecolor='black',
            alpha=0.7,
            label=cluster_name if i == 0 else None  # Add legend label only once
        )

# Add labels for each feature
ax.set_xticks(angles[:-1])
ax.set_xticklabels(Median_df.columns)

# Add labels for radial axis with adjusted range
ax.set_yticks(np.linspace(min_value, max_value, num=7))
ax.set_yticklabels([f"{v:.1f}" for v in np.linspace(min_value, max_value, num=7)], color="grey", size=10)
ax.set_ylim(min_value, max_value)

# Add title and legend
ax.set_title('Spider Plot with Radial Bars', size=16)
ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))

#plt.savefig('C:/Users/E14Ge/Desktop/Data_results/UMAP_clusters_comparison/Radar_plot.svg')
#plt.savefig('C:/Users/E14Ge/Desktop/Data_results/UMAP_clusters_comparison/Radar_plot.pdf')
#plt.savefig('C:/Users/E14Ge/Desktop/Data_results/UMAP_clusters_comparison/Radar_plot.png',dpi=600)


# Show the plot
plt.show()




""" FEATURE CLASSIFICATION """


# I run Random Forest algorithm to get feature importance
# and to quantify which feature best distinguish one cluster
#from the others

# According to how many clusters are foud the feature 
# importance is showed as a heatmap of a barplot
# In general: if #Clusters <= 2 then barplot
# if #Clusters > 2 then heatmap


if n_clusters <= 2:

    # Perform Rabdom Forest to asses the accuracy
    
    # Separate features and target variable
    X = reduced_df.iloc[:, :-1]  # All columns except the last one
    y = reduced_df.iloc[:, -1]   # The last column (subpopulation labels)
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Train the Random Forest Classifier
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    
    # Evaluate the model on the test set
    y_pred = rf.predict(X_test)
    print("Accuracy on Test Set:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    
    
    
    # Run the Random Forest on the whole dataset to get feature 
    # importance
    
    rf = RandomForestClassifier(random_state=42)
    rf.fit(reduced_df.iloc[:, :-1],labels)
    
    feature_importances = pd.Series(rf.feature_importances_, index=reduced_df.columns[:-1])
    print("Feature Importances:\n", feature_importances.sort_values(ascending=False))
    
    # Feature importance analysis
    importances = rf.feature_importances_
    
    # Create a DataFrame for feature importance
    feature_importance_df = pd.DataFrame({
        "Feature": X.columns,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)
    
    # Display feature importance
    print("\nFeature Importance:\n", feature_importance_df)
    
    
    plt.figure(figsize=(8, 6))
    plt.barh(feature_importance_df["Feature"], feature_importance_df["Importance"], color='Green')
    plt.xlabel('Feature importance (a.u.)')
    #plt.ylabel('Row Index')
    #plt.title('Ranking of Rows by Sum of Squares')
    plt.gca().invert_yaxis()  # Highest rank at the top
    plt.xlim([0,1])
    #plt.xticks(range(0, 0.5, 0.1)) 
    
    #plt.savefig('C:/Users/E14Ge/Desktop/Data_results/UMAP_clusters_comparison/Feat_importanceAll.svg')
    #plt.savefig('C:/Users/E14Ge/Desktop/Data_results/UMAP_clusters_comparison/Feat_importanceAll.pdf')
    #plt.savefig('C:/Users/E14Ge/Desktop/Data_results/UMAP_clusters_comparison/Feat_importanceAll.png',dpi=600)
    
    plt.show()
    
else:
    
    
    # feature importance as heatmap
    
    # Separate features and target variable
    X = reduced_df.iloc[:, :-1]  # All columns except the last one
    y = reduced_df.iloc[:, -1]   # The last column (subpopulation labels)
    
    # Initialize the LabelBinarizer to create binary labels for one-vs-all classification
    lb = LabelBinarizer()
    y_bin = lb.fit_transform(y)
    
    # Store feature importance for each cluster
    feature_importances = []
    
    for i, cluster in enumerate(lb.classes_):
        # Binary label for current cluster
        y_binary = y_bin[:, i]
    
        # Train Random Forest Classifier
        rf = RandomForestClassifier(random_state=42, n_estimators=100)
        rf.fit(X, y_binary)
    
        # Store feature importances
        feature_importances.append(rf.feature_importances_)
    
    features_name = X.columns.tolist()
    
    # Convert to DataFrame for better handling
    importance_df = pd.DataFrame(
        feature_importances, 
        index=[f"Cluster {cluster}" for cluster in lb.classes_], 
        columns=X.columns#[feature for feature in X.columns]#[f"Feature {i+1}" for i in range(X.shape[1])]
    )
    
    # Plot heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(importance_df, cmap="Greens", annot=False, cbar=True, vmax=1, vmin=0,linecolor="grey",linewidths=0.5)
    plt.title("Feature Importance Heatmap")
    plt.xlabel("Features")
    plt.ylabel("Clusters")
    plt.tight_layout()
    
    #plt.savefig('C:/Users/E14Ge/Desktop/Data_results/UMAP_clusters_comparison/Feat_importanceHM.svg')
    #plt.savefig('C:/Users/E14Ge/Desktop/Data_results/UMAP_clusters_comparison/Feat_importanceClusterHM.pdf')
    #plt.savefig('C:/Users/E14Ge/Desktop/Data_results/UMAP_clusters_comparison/Feat_importanceClusterHM.png',dpi=600)
    
    plt.show()


























