import numpy as np



def cleaning(matrix):
    
    # This function takes a matrix of features and returns the same matrix 
    # cleaned of NAN values.
    
    # Identify rows with NaN values
    NANrows = np.isnan(matrix).any(axis=1)
    
    # Filter out rows with NaN values
    matrix = matrix[~NANrows]
    
    return matrix


def ecdf(data):
    
    """ 
    Compute the Empirical Cumulative Distribution Function (ECDF) for 
    a given dataset.
    F(X) = P(X <= x) -> Sort data and then the probability to find a value
    lower than x is equal to the total number of values before x divided the
    total number in the dataset. The total number of values before x is 
    represented by the position of x value (np.arange()) in the sorted array 
    for values.
    
    """
    n = len(data)
    x = np.sort(data)
    y = np.arange(1, n+1) / n
    return x, y

def two_sample_signed_ks_statistic(sample1, sample2):
    
    """
    Compute the two-sample signed KS statistic.
    
    Step 1: Compute ECDFs for Both Samples: For each sample, sort the data 
    and compute the cumulative probabilities. This gives the ECDF for each 
    sample.
    
    Step 2: Combine and Sort All Unique x-values: Merge the sorted values 
    from both samples into a single sorted array of unique x-values. 
    This combined set represents all potential points where the ECDFs might 
    change.
    
    Step 3: Evaluate (Interpolate) ECDFs at Combined x-values: Determine the 
    value of each ECDF at every point in the combined x-values array. This 
    involves finding where each x-value from the combined set falls within 
    each sample's sorted values and computing the corresponding ECDF value.
    
    Return: KS value and signed KS value 
    
    """
   
    # Compute ECDF for both samples
    x1, y1 = ecdf(sample1)
    x2, y2 = ecdf(sample2)
    
    # Find the union of both sets of x-values
    all_x = np.sort(np.concatenate((x1, x2)))
    
    # Interpolate the ECDFs on the combined x-values
    # For each value in all_x, np.searchsorted finds the index where this 
    # value would fit in the sorted ECDF arrays (x1 and x2). This index divided
    #by the length of the array gives the ECDF value.
    y1_interp = np.searchsorted(x1, all_x, side='right') / len(sample1)
    y2_interp = np.searchsorted(x2, all_x, side='right') / len(sample2)
    
    # Compute the signed KS statistic
    # np.argmax() find the indices of the maximum values along a specified 
    # axis.
    differences = y1_interp - y2_interp
    ks_statistic = np.max(np.abs(differences))
    signed_ks_statistic = differences[np.argmax(np.abs(differences))]
    
    return ks_statistic, signed_ks_statistic


def FilterFeat(df, threshold):
    
    """
   Removes highly correlated features from a DataFrame, retaining the one with higher variance.
   
   Args:
       df (pd.DataFrame): Input DataFrame with features as columns.
       threshold (float): Correlation threshold above which features are considered redundant.
       
   Returns:
       pd.DataFrame: DataFrame with redundant features removed.
       
   """
    
   # Compute correlation matrix
    correlation_matrix = df.corr()
        
    # Find feature pairs with correlation above the threshold (not duplicate feature pairs )
    correlated_pairs = []
    for i, feature1 in enumerate(correlation_matrix.columns):
        for j, feature2 in enumerate(correlation_matrix.columns):
            if i < j and abs(correlation_matrix.loc[feature1, feature2]) > threshold:
                correlated_pairs.append((feature1, feature2, abs(correlation_matrix.loc[feature1, feature2])))


    # Sort correlated pairs by descending correlation
    correlated_pairs = sorted(correlated_pairs, key=lambda x: x[2], reverse=True)
       
    removed_features = set()# set of unordered unique elements
    
    # remove features with lower variance
    for feature1, feature2, _ in correlated_pairs:
        
        # Skip the pair if one of the features is already removed
        if feature1 in removed_features or feature2 in removed_features:
            continue
        
        # Compare variances to decide which feature to remove
        if df[feature1].var() > df[feature2].var():
            removed_features.add(feature2)
        else:
            removed_features.add(feature1)

    # Return the DataFrame with redundant features removed
    return df.drop(columns=list(removed_features))
    