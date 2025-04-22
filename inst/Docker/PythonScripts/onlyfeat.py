import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import MyFunctions2 as F 
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
import seaborn as sns
import umap
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelBinarizer
from itertools import combinations
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
import sys

sys.path.append('PythonScripts')
import MyFunctions2 as F 

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


# compute the correlation matrix of the original features
correlation_matrixNEW = reduced_df.corr()

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




print("Clutering")

""" CLUSTERING """

# Optimal k is where the silhouette score is maximized
#optimal_k = k_range[silhouette_scores.index(max(silhouette_scores))]
optimal_k = 3
print(optimal_k)
print("ciao")


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


""" COMPARISON OF FEATURES BETWEEN CLUSTERS """



# VIOLIN PLOTS
reduced_df['Clusters'] = labels

# Get list of feature columns (excluding the cluster column)
features = [col for col in reduced_df.columns if col != 'Clusters']


""" FEATURE CLASSIFICATION """


# I run Random Forest algorithm to get feature importance
# and to quantify which feature best distinguish one cluster
#from the others

# According to how many clusters are foud the feature 
# importance is showed as a heatmap of a barplot
# In general: if #Clusters <= 2 then barplot
# if #Clusters > 2 then heatmap

print(reduced_df.head())
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
    plt.savefig('/Users/simonepernice/Desktop/res/Feat_importanceAll.png',dpi=600)
    
    #plt.show()
    
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
    plt.savefig('/Users/simonepernice/Desktop/res/Feat_importanceClusterHM.png',dpi=600)
    
    plt.show()


























