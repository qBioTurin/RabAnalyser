import os
from tifffile import imread
import numpy as np
from skimage import exposure
import matplotlib.pyplot as plt
from skimage.measure import regionprops
import MyFunctions_FeatExtract as F
from skimage.morphology import binary_erosion
import time
import pandas as pd

# Record the start time
start_time = time.perf_counter()


# path where images and experimental conditions are stored
# Esp_RabAnalyser is the folder containing the folders with the
# experimental condition. Within each of this folders there are sub-folders
# containing the input for the script
root_dir = r"C:/Users/E14Ge/Desktop/Nuova cartella/bfmatlab/Esp_RabAnalyser/prova"

# List only folders (i.e., experimental conditions) and exclude hidden ones
condition_folders = [
    name for name in os.listdir(root_dir)
    if os.path.isdir(os.path.join(root_dir, name)) and not name.startswith('.')
]


""" LOOP on each experimental condition """

for condition_name in condition_folders:
    
    #condition_name = 'CET_Rab5'
    
    
    condition_path = os.path.join(root_dir, condition_name)
    print(f"Processing condition: {condition_name}")

    # Initialize feature matrix for Rab spots
    Clusters = []

    print("Creating image datastores")

    
    # Load image paths for each required input
    dsnucleus_mask = F.load_images(condition_path,"nucleus_mask") # nucleus_mask
    dscell_m = F.load_images(condition_path,"cell_mask") # cell mask
    dsrab = F.load_images(condition_path,"rab5")         # Rab raw signal
    dsspot = F.load_images(condition_path,"rab5_mask")   # mask of Rab spots
    

    # Check all datastores have the same number of images
    
    # n_images rappressenta il numero di immagini che ci sono per condizione
    # sperimentale. Darei la possibilità all'utente di scegliere se analizzare 
    # tutte le immagini oppure solo un sottoinsieme.
    
    n_images = min(len(dsnucleus_mask), len(dscell_m), len(dsrab), len(dsspot)) #len(dscet))


    # cell label
    eth = 1       
    
    # Loop over each image field
    for k in range(n_images):
        print(f"Processing image {k + 1}/{n_images}")
        
        #k = 0
        # --- Load nucleus mask ---
        nucleus_m = imread(dsnucleus_mask[k])  # 2D array
        # Optional: display
        # plt.figure(figsize=(8, 5)),plt.imshow(nucleus_m, cmap='gray'); plt.title("Nucleus Mask"); plt.show()
    
        # --- Load cell mask ---
        cell_m = imread(dscell_m[k])
        # plt.figure(figsize=(8, 5)),plt.imshow(cell_m, cmap='gray'); plt.title("Cell Mask"); plt.show()
    
        # --- Load Rab raw image ---
        rab = imread(dsrab[k])
        rabg = exposure.rescale_intensity(rab, in_range='image', out_range=(0.0, 1.0))  # like mat2gray
        # plt.imshow(rabg, cmap='gray'); plt.title("Rab signal"); plt.show()
    
        # --- Load Rab spot mask ---
        spot = imread(dsspot[k])
        # plt.imshow(spot, cmap='gray'); plt.title("Rab spot mask"); plt.show()
    
        # --- Load condition image (e.g., Cetuximab) ---
        #cond = imread(dscet[k])
        #condg = exposure.rescale_intensity(cond, in_range='image', out_range=(0.0, 1.0))
        # plt.imshow(condg, cmap='gray'); plt.title("Condition image"); plt.show()
        
        
        print('nucleus and cell synchronization');
        print(k);
            
        # In CellProfiler I have to keep in consideration each nucleus as a seed to
        # segment the corresponding cell. However, when I segment single cells I
        # discard those attached to the edges of image. Cells attacched to edges of
        # image are croppend, so not useful for analysis. Therefore, I expect to
        # have more nuclei than cells, as cells can be discarded but nuclei not.
        # Here, I re-map labels of nuclei and single cells in order to have nucleus 
        # with the same label of the cell.
        
        # Get centroids of nuclei
        nucleus_struct = regionprops(nucleus_m)
        nucleus_cntrd = np.array([prop.centroid for prop in nucleus_struct])
        
        # Create new nucleus mask, same shape
        nucleus_m_new = np.zeros_like(nucleus_m, dtype=np.int32)
        
        # Keep only nuclei whose centroids fall within segmented cells
        for centroid in nucleus_cntrd:
            
            x,y = np.round(centroid).astype(int)  # row -> x, column -> y
        
            if cell_m[x, y] != 0:
                cell_label = cell_m[x, y]
                nucleus_label = nucleus_m[x, y]
        
                # Find all pixels belonging to this nucleus
                nucleus_pixels = np.argwhere(nucleus_m == nucleus_label)
        
                for xpix, ypix in nucleus_pixels:
                    nucleus_m_new[xpix, ypix] = cell_label
                    
        #plt.figure(figsize=(8, 5)),plt.imshow(cell_m, cmap='gray'); plt.title("Nucleus Mask"); plt.show()
        #plt.figure(figsize=(8, 5)),plt.imshow(nucleus_m_new, cmap='gray'); plt.title("Nucleus Mask"); plt.show()
        #plt.figure(figsize=(8, 5)),plt.imshow(nucleus_m, cmap='gray'); plt.title("Nucleus Mask"); plt.show()
        
        # Extract updated centroids from new nucleus mask
        nucleus_struct_new = regionprops(nucleus_m_new)
        nucleus_cntrd_new = np.array([prop.centroid for prop in nucleus_struct_new])
        
        cell_struct_new = regionprops(cell_m)
        cell_cntrd_new = np.array([prop.centroid for prop in cell_struct_new])
        
        
        
        # Proceed only if there are cells to be analyzed in the image k
        if nucleus_cntrd_new.shape[0] > 0:
            
            max_cell = np.max(cell_m)
            
            
            for L in range(1, max_cell + 1):  # cell labels start from 1
            
                print(f"\nCell: {L}, Image: {k}\n")
                
                temp = []
                
                #L = 1
                # Create a new image with only the current cell
                mask_cell = (cell_m == L)
                
                #coords = np.argwhere(mask_cell)
                #coords_label_1 = np.argwhere(cell_m == 1)
                
                #if not np.any(mask_cell):
                    #continue  # skip if this cell doesn't exist
        
                # Apply the same mask to the Rab spot mask
                spotnew = spot.copy()
                spotnew[~mask_cell] = 0
        
                cell_m_new = cell_m.copy()
                cell_m_new[~mask_cell] = 0
        
                # Filter Rab spots smaller than n pixels.
                # To have reliable measurements (mostly about shape) spots with
                # more than 5-8 pixels are better.
                
                # PARAMETRO DA LASCIAR SETTARE ALL'UTENTE
                n = 1
        
                # find area of the spots within the cell of interest
                spot_props = regionprops(spotnew)
                       
                # Build list of the labels of spots to remove
                small_labels = [prop.label for prop in spot_props if prop.area < n]
        
                # Remove spots smaller than n pixels
                spotf = spotnew
                for lbl in small_labels:
                    spotf[spotf == lbl] = 0
        
                
######################### Rab SPOTS SIZE ######################################    
                

                print('SIZE')
 
                # Extract Rab spot area from spotf
                area_props = regionprops(spotf)
                area = np.array([prop.area for prop in area_props])
                
                # I add for each spot measumets the label (eth) of the cells to 
                # which it belongs
                # I generate an array with same dimension of area filled with 
                #eth value
                eth_column = np.full((area.shape[0], 1), eth)
                
                # Reshape area to a column
                area_column = area.reshape(-1, 1)
                
                # Concatenate horizontally
                area = np.hstack((eth_column, area_column))  
                
                
                
                         
######################## Rab SPOTS INTEGRAL INTENSITY #########################               
                
                print('INTEGRAL INTENSITY')

                # Get region properties, including intensity values
                Intsum_props = regionprops(spotf, intensity_image=rabg)  
                
                # Extract integral intensity for each region
                Intsum = np.array([np.sum(prop.intensity_image) for prop in Intsum_props ])
                
                # Horizontally concatenate with previous features
                temp = np.hstack((area, Intsum[:,None]))
                
                
######################## Rab SPOTS MEAN INTENSITY #############################

                print('MEAN INTENSITY')

                # Get region properties, including intensity values
                MeanInt_props = regionprops(spotf, intensity_image=rabg)
                
                # Extract mean intensity for each region
                MeanInt = np.array([prop.mean_intensity for prop in MeanInt_props])
                
                # Horizontally concatenate with previous features 
                temp = np.hstack((temp, MeanInt[:,None]))
                
                
                
######################### Rab SPOTS CIRCULARITY ###############################               
                
                print('CIRCULARITY')
                
              
                # Extract circularity from each spots
                Circ = F.circularity(spotf)
                
                # Horizontally concatenate with previous features 
                temp = np.hstack((temp, Circ[:,None]))
                
                
######################## Rab SPOTS ECCENTRICITY ###############################

                print('ECCENTRICITY')
                
                
                # Extract properties from each spots
                ecc_props = regionprops(spotf)
                
                # Extract eccentricity for each region
                ecc = np.array([prop.eccentricity for prop in ecc_props])
                
                # Horizontally concatenate with previous features 
                temp = np.hstack((temp, ecc[:,None]))

                  
######################## Rab SPOTS ELONGATION #################################               
                
                
                print('ELONGATION')
                
                # Extract properties from each spots
                elo_props = regionprops(spotf)
                
                elo = []
                for p in elo_props:
                    minor = p.minor_axis_length
                    major = p.major_axis_length
            
                    if minor > 0:
                        elong = major / minor
                    else:
                        elong = np.nan  # avoid division by zero
            
                    elo.append(elong)
                elo = np.array(elo)
                    
                # Horizontally concatenate with previous features 
                temp = np.hstack((temp, elo[:,None]))
                            
                
                #uniqueVals = np.unique(cell_m_new)
                
                
               
######################## Rab SPOTS DISTANCE FROM NUCLEUS ######################  

                print('DISTANCE FROM NUCLEUS')          
                
                props_cell = regionprops(cell_m_new)
                
                # distance from nucleus centroid are normalized by the major 
                # diagonal of the smaller rectangulus enclosing the cell
                # Bounding box format: (min_row, min_col, height, width)
                minr, minc, maxr, maxc = props_cell[0].bbox
                height = maxr - minr
                width = maxc - minc
                diag = np.sqrt(width**2 + height**2)
                
                # Get nucleus centroid (of label L)
                props_nuc = regionprops(nucleus_m_new)
                nuc_centroids = [p.centroid for p in props_nuc]
                
                # nucleus centroids are orderd according to nucleus labels. Thus,
                # nuclei with label "1" are in postion 0 in nuc_centroids 
                # variable, nuclei with label "2" are in postion 1 in 
                # nuc_centroids variable and so on
                if L - 1 < len(nuc_centroids):
                    nuc_centroid = nuc_centroids[L - 1]  
                else:
                    raise IndexError(f"Nucleus label {L} not found")
                
                # Compute distances of Rab clusters to nucleus centroid (custom function)
                Dist = F.NucD(nuc_centroids[L-1], diag, spotf)
                
                # Horizontally concatenate with previous features 
                temp = np.hstack((temp, Dist[:,None]))
                
                
################### Rab SPOTS DISTANCE FROM PLASMA MEMBRANE ###################             
                
                print('DISTANCE FROM PLASMA MEMBRANE')
                
                # I binarize the matrix identifiyng the cell of interest
                BWcell = (cell_m_new > 0).astype(np.uint8)
                
                # Get plasma membrane (perimeter of the cell).
                # ^ XOR
                cellPerim = BWcell ^ binary_erosion(BWcell)
                
                #plt.figure(figsize=(8, 5)),plt.imshow(cellPerim, cmap='gray'); plt.title("Nucleus Mask"); plt.show()
                
                # Measure the distance from plasma membrane for each Rab spot
                Pdist = F.PMd(cellPerim, diag, spotf)
                
                # Horizontally concatenate with previous features 
                temp = np.hstack((temp, Pdist[:,None]))
                
                
                
################### DISTANCE TO THE NEAREST Rab SPOT ##########################
                
                print('DISTANCE TO THE NEAREST Rab SPOT')


                SpotDNN = F.DNN(diag,spotf)
                
                # Horizontally concatenate with previous features 
                temp = np.hstack((temp, SpotDNN[:,None]))
                


################# MOST FREQUENT DISTANCE BETWEEN Rab SPOTS ####################
                
                print('MOST FREQUENT DISTANCE BETWEEN Rab SPOTS')

                SpotDist = F.DistSpot(diag,spotf)
                
                # Horizontally concatenate with previous features 
                temp = np.hstack((temp, SpotDist[:,None]))

                

               
################### NUMBER OF THE NEIGHBOURING Rab SPOTS ######################

                print('NUMBER OF THE NEIGHBOURING Rab SPOTS')

                # radius to search for neighboring clusters. The unit of measure
                # is in pixels. This number should be set considering the 
                # microscope resolution, pixel size and the tipical distance/size 
                # between endosomes (early endosomes span from 500 nm to 1000 nm)
                
                # PARAMETRO DA LASCIAR SETTARE ALL'UTENTE
                d = 10


                NumSpotD = F.NumNeighb(d,spotf,cell_m_new);
                
                # Horizontally concatenate with previous features 
                temp = np.hstack((temp, NumSpotD[:,None]))
                
                
                
                # Add Rab clusters measurements to final data matrix
                Clusters.append(temp)
        
                # increase the cell counting (cell label)
                eth = eth + 1;  

    Clusters = np.vstack(Clusters)


    """ DATAFRAME CREATION """
   
    # Create new dataframe where the colums contain the name of the feature measured
   
    print("\nDataframe creation")
   
    # Load the column names from the Excel file
    # The feature names are listed in an excel file
    excel_file = 'Feature_extraction_labels.xlsx'
    column_names_df = pd.read_excel(excel_file, sheet_name='Sheet1', header=None)
   
    # Extract the column names from the first row
    column_names = column_names_df.iloc[0].tolist()
   
    # Ensure the length of column_names matches the number of columns in your data matrix
    # 12
    num_columns = Clusters.shape[1]
    if len(column_names) != num_columns:
        raise ValueError("Number of column names in Excel file does not match the number of columns in the data matrix")
   
    # Convert the NumPy array to a Pandas DataFrame with the loaded column names
    df = pd.DataFrame(Clusters, columns=column_names)
    
    
    """ SAVING """
    
    # Construct the full path and filename
    save_name = os.path.join(root_dir, f"{condition_name}.xlsx")
    
    # Save the DataFrame to Excel
    df.to_excel(save_name, index=False)



