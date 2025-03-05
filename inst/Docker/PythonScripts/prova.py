import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import MyFunctions2 as F 
import seaborn as sns
import sys

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

correlation_matrixOLD.to_excel('./prova.xlsx',index=False) 

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

plt.savefig('allTRTDRab5UMAP.png',dpi=600)

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

plt.savefig('provaFIltered.png',dpi=600)
correlation_matrixNEW.to_excel('./provaFIltered.xlsx',index=False) 
