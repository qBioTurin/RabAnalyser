import numpy as np
from matplotlib.colors import LinearSegmentedColormap



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



def make_whiteband_cmap(
    left_colors,                 # list of colors for the left side (vmin -> white_min)
    right_colors,                # list of colors for the right side (white_max -> vmax)
    vmin, vmax,                  # data range used for normalization
    white_min, white_max,        # interval to render as white
    white_color=(1, 1, 1, 1),    # RGBA or named color for the band
    n=256                        # total colormap resolution
):
    """
    Build a colormap that is:
      - left_colors from vmin .. white_min
      - WHITE from white_min .. white_max
      - right_colors from white_max .. vmax
    """
    # Normalize band edges into [0,1]
    a = (white_min - vmin) / (vmax - vmin)
    b = (white_max - vmin) / (vmax - vmin)
    a, b = sorted((a, b))
    a = float(np.clip(a, 0, 1))
    b = float(np.clip(b, 0, 1))

    # Allocate number of steps per segment
    N_left  = max(2, int(n * a))             # [0, a)
    N_mid   = max(1, int(n * (b - a)))       # [a, b] -> white band
    N_right = max(2, int(n * (1 - b)))       # (b, 1]

    # Build left gradient ending at white (exclude the final white sample to avoid duplication)
    cmap_left = LinearSegmentedColormap.from_list("left", left_colors + [white_color])
    colors_left = [cmap_left(x) for x in np.linspace(0, 1, N_left, endpoint=False)]

    # Middle white band
    colors_mid = [white_color] * N_mid

    # Build right gradient starting from white (drop the first white sample to avoid duplication)
    cmap_right = LinearSegmentedColormap.from_list("right", [white_color] + right_colors)
    xs_right = np.linspace(0, 1, N_right + 1, endpoint=True)[1:]  # skip 0 (white)
    colors_right = [cmap_right(x) for x in xs_right]

    colors = colors_left + colors_mid + colors_right
    
    return LinearSegmentedColormap.from_list("custom_white_band", colors, N=len(colors))
