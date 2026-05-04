#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import glob
import pandas as pd
import nibabel as nib
from nilearn import plotting
import matplotlib.pyplot as plt


data_dir = './data/1160600'

# --- 1. READING THE CLINICAL FILE ---
csv_file = os.path.join(data_dir, 'cobre_model_group.csv')
if os.path.exists(csv_file):
    df_labels = pd.read_csv(csv_file)
    print("Overview of the clinical data(first 5 rows):")
    display(df_labels.head())
    display(df_labels.shape)
    display(df_labels.columns)
    display(df_labels.info())
    df_labels.columns = df_labels.columns.str.strip()
    print("les valeurs sont: \n")  # Remove any leading/trailing whitespace from column names
    print(df_labels['sz'].value_counts())
    display(df_labels.iloc[72:])

else:
    print("Warning: csv file not found.")



# In[2]:


# --- 2. select first subject and sanity check ---
patient_files = glob.glob(os.path.join(data_dir, '*.nii.gz'))
print(f"\n Number of irmF images found: {len(patient_files)}")

if len(patient_files) > 0:
    test_file = patient_files[0]
    print(f" test file selected: {os.path.basename(test_file)}")

    # --- 3. sanity check: load and visualize the first image ---
    print("\nLoading the first image for sanity check...")
    img = nib.load(test_file)
    print(f" Dimensions of 4D matrix (X, Y, Z, T): {img.shape}")

    # we extract the first image from the film (time point 0)
    first_volume = img.slicer[..., 0]

    # we display the brain with nilearn
    plotting.plot_epi(first_volume, 
                      title="brain of patient ( t=0)", 
                      display_mode='z', #horizontal slices 
                      cut_coords=5, # number of slices to display
                      cmap='gray') # gray levels
    plt.show()
else:
    print("No patient files found in the data directory.")


# In[3]:


# --- 4. Masking preparation (schaefer atlas 1000) ---
import numpy as np
from nilearn import image, datasets
from nilearn.maskers import NiftiLabelsMasker

print("\nLoading the Schaefer atlas (1000 parcels)...")

dset_schaefer_1000 = datasets.fetch_atlas_schaefer_2018(n_rois=1000, resolution_mm=2)
dset_ho = datasets.fetch_atlas_harvard_oxford('sub-maxprob-thr25-2mm') #21 rois


#load the harvard oxford atlas and create masker
if not isinstance(dset_ho.maps, str):
    original_img = dset_ho.maps
else:
    import nibabel as nib
    original_img = nib.load(dset_ho.maps)

clean_data = original_img.get_fdata().copy()



# delete region 1 and 2 (Cortex/White Matter global)
# set to 0 every pixels that belong to these labels. 
clean_data[clean_data == 1] = 0
clean_data[clean_data == 2] = 0

#create new clean atlas image 
cleaned_img = image.new_img_like(original_img, clean_data)

# MASKERS

masker_schaefer = NiftiLabelsMasker(labels_img=dset_schaefer_1000.maps, 
#standadize with z-score normalization to avoid biais in the glasso penality, and due to the arbitrary units of fmri
                                    standardize='zscore_sample', 
                                    memory='nilearn_cache', verbose=0)
#same arguments for the subcortical atlas
masker_subcort = NiftiLabelsMasker(labels_img=cleaned_img,
                                    standardize='zscore_sample',
                                    memory='nilearn_cache', verbose=0)
print("Atlases loaded and maskers created. (Schaefer 1000 parcels and Harvard-Oxford subcortical atlas 21 parcels)")


# In[4]:


import numpy as np
import h5py #for database storage
from sklearn.covariance import GraphicalLassoCV
from tqdm.notebook import tqdm
from nilearn.connectome import ConnectivityMeasure
from sklearn.preprocessing import StandardScaler
import os

# Initialization of lists to store BOTH matrices
hc_pearson_list = []
scz_pearson_list = []
hc_glasso_list = []
scz_glasso_list = []

print(f"Start processing for {len(patient_files)} patients...")

# 2. Configure BOTH algorithms
correlation_measure = ConnectivityMeasure(kind='correlation')
glasso = GraphicalLassoCV(cv=3, n_jobs=-1, max_iter=1000, tol=1e-2)

# loop over all .nii.gz files
for file_path in tqdm(patient_files, desc="IRM processing"):
    try:
        # Time-serie extraction (time x (1000 ROI + 21 subcortical ROIs))
        ts_schaefer = masker_schaefer.fit_transform(file_path)
        ts_subcort = masker_subcort.fit_transform(file_path)

        # Concatenate the time series from both atlases
        time_series = np.hstack([ts_schaefer, ts_subcort])

        # --- ROBUST SECURITY ---
        time_series = np.nan_to_num(time_series, nan=0.0)

        # handling dead columns (null variance)
        std_devs = np.std(time_series, axis=0)
        dead_cols = std_devs < 1e-6

        if np.any(dead_cols):
            # jitter injection to stabilize the math
            noise = np.random.normal(0, 1e-7, size=(time_series.shape[0], np.sum(dead_cols)))
            time_series[:, dead_cols] = noise

        if not np.isfinite(time_series).all():
            print(f"Skipping {os.path.basename(file_path)} due to NaN or inf values in the time series.")
            continue

        scaler = StandardScaler()
        time_series = scaler.fit_transform(time_series)

        # SÉCURITÉ SUPPLÉMENTAIRE : On re-nettoie après le scaler
        time_series = np.nan_to_num(time_series, nan=0.0)

        # 3. COMPUTE PEARSON
        pearson_matrix = correlation_measure.fit_transform([time_series])[0]
        np.fill_diagonal(pearson_matrix, 0) # 0 initialisation of the diagonal

        # 4. COMPUTE GRAPHICAL LASSO
        glasso.fit(time_series)
        glasso_matrix = glasso.precision_ # Extract the precision matrix
        np.fill_diagonal(glasso_matrix, 0) # 0 initialisation of the diagonal

        # 5. Sort patient (HC vs SCZ)
        file_name = os.path.basename(file_path)
        if 'contxxx' in file_name:
            hc_pearson_list.append(pearson_matrix)
            hc_glasso_list.append(glasso_matrix)
        elif 'szxxx' in file_name:
            scz_pearson_list.append(pearson_matrix)
            scz_glasso_list.append(glasso_matrix)

    except Exception as e:
        print(f"Error with file {file_path}: {e}")

# Convert lists to "cubes" (3D numpy arrays)
hc_pearson_array = np.array(hc_pearson_list)
scz_pearson_array = np.array(scz_pearson_list)
hc_glasso_array = np.array(hc_glasso_list)
scz_glasso_array = np.array(scz_glasso_list)

print(f"\nCalculation completed! Final dimensions:")
print(f"Healthy controls (Pearson/Glasso): {hc_pearson_array.shape} / {hc_glasso_array.shape}")
print(f"Schizophrenia patients (Pearson/Glasso): {scz_pearson_array.shape} / {scz_glasso_array.shape}")

# ---------------------------------------------------
# CREATION OF A DATABASE WITH HDF5 (DATA PERSISTENCE)
# ---------------------------------------------------

# Rename DB to reflect combined contents
db_path = 'cobre_combined_connectomes_database.h5'

with h5py.File(db_path, 'w') as db:
    # Creating datasets for Pearson
    db.create_dataset('hc_pearson', data=hc_pearson_array, compression='gzip')
    db.create_dataset('scz_pearson', data=scz_pearson_array, compression='gzip')

    # Creating datasets for GLasso
    db.create_dataset('hc_glasso', data=hc_glasso_array, compression='gzip')
    db.create_dataset('scz_glasso', data=scz_glasso_array, compression='gzip')

    # Adding metadata
    db.attrs['description'] = 'Combined Database (Pearson & GLasso) for COBRE dataset'
    db.attrs['number_of_hc'] = hc_pearson_array.shape[0]
    db.attrs['number_of_scz'] = scz_pearson_array.shape[0]

print(f"\nDatabase created successfully at: {db_path} !")


# In[5]:


import h5py
import matplotlib.pyplot as plt

db_path = 'cobre_combined_connectomes_database.h5'

with h5py.File(db_path, 'r') as db:
    print("Keys in the database:", list(db.keys()))

    # Load Patient 0 from all four datasets
    hc_pearson = db['hc_pearson'][0]
    hc_glasso = db['hc_glasso'][0]
    scz_pearson = db['scz_pearson'][0]
    scz_glasso = db['scz_glasso'][0]

# Set up a 2x2 grid for plotting
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# 1. HC - Pearson
im1 = axes[0, 0].imshow(hc_pearson, cmap='RdBu_r', vmin=-1, vmax=1)
axes[0, 0].set_title("Patient 0 (HC) - Pearson Correlation", fontweight='bold')
fig.colorbar(im1, ax=axes[0, 0], fraction=0.046, pad=0.04)

# 2. HC - GLasso
im2 = axes[0, 1].imshow(hc_glasso, cmap='RdBu_r')
axes[0, 1].set_title("Patient 0 (HC) - GLasso Precision", fontweight='bold')
fig.colorbar(im2, ax=axes[0, 1], fraction=0.046, pad=0.04)

# 3. SCZ - Pearson
im3 = axes[1, 0].imshow(scz_pearson, cmap='RdBu_r', vmin=-1, vmax=1)
axes[1, 0].set_title("Patient 0 (SCZ) - Pearson Correlation", fontweight='bold')
fig.colorbar(im3, ax=axes[1, 0], fraction=0.046, pad=0.04)

# 4. SCZ - GLasso
im4 = axes[1, 1].imshow(scz_glasso, cmap='RdBu_r')
axes[1, 1].set_title("Patient 0 (SCZ) - GLasso Precision", fontweight='bold')
fig.colorbar(im4, ax=axes[1, 1], fraction=0.046, pad=0.04)

plt.suptitle("Functional Connectomes: Algorithm & Cohort Comparison", fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()


# In[6]:


from nilearn import plotting, image
import matplotlib.pyplot as plt

# 1. chosen patient file for visualization (first one in the list (default))
sample_patient = patient_files[0] # we can change the index to visualize another patient if we want

print(f"file verification: {sample_patient}")

#2. load the image (mean accross time for better visualization)
mean_img = image.mean_img(sample_patient)

img_schaefer = masker_schaefer.labels_img_
img_subcort = masker_subcort.labels_img_

#3. figure configuration
fig, axes = plt.subplots(2, 1, figsize=(12, 10))

#visualisation of the schaefer atlas (cortex)
plotting.plot_roi(img_schaefer, bg_img=mean_img,
                  title="Superposition schaefer atlas", 
                  display_mode='ortho', cut_coords=(0, 0, 0), 
                  axes=axes[0])

# visualisation of the subcortical atlas
plotting.plot_roi(img_subcort, bg_img=mean_img, 
                  title="Superposition Subcortical Atlas", 
                  display_mode='ortho', cut_coords=(0, 0, 0), 
                  axes=axes[1])

plt.show()


# In[ ]:




