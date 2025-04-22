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



# VIOLIN PLOTS
reduced_df['Clusters'] = labels

# Get list of feature columns (excluding the cluster column)
features = [col for col in reduced_df.columns if col != 'Clusters']


    
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


























