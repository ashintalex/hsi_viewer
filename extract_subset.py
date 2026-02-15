#!/usr/bin/env python3
"""Extract a subset/crop from a hyperspectral TIFF image"""

import numpy as np
import tifffile
import h5py
import sys
from pathlib import Path

def extract_subset(input_path, output_path, row_start, row_end, col_start, col_end):
    """
    Extract a spatial subset from a hyperspectral image
    
    Args:
        input_path: Path to input TIFF file
        output_path: Path to output TIFF file
        row_start: Starting row index
        row_end: Ending row index (exclusive)
        col_start: Starting column index
        col_end: Ending column index (exclusive)
    """
    print(f"Loading: {input_path}")
    data = tifffile.imread(input_path)
    print(f"Original shape: {data.shape}")
    print(f"Data type: {data.dtype}")
    
    # Extract subset
    if data.ndim == 3:
        # Assume (rows, cols, bands) format
        subset = data[row_start:row_end, col_start:col_end, :]
    elif data.ndim == 2:
        subset = data[row_start:row_end, col_start:col_end]
    else:
        raise ValueError(f"Unsupported number of dimensions: {data.ndim}")
    
    print(f"Subset shape: {subset.shape}")
    print(f"Subset region: rows [{row_start}:{row_end}], cols [{col_start}:{col_end}]")
    print(f"Data range: {np.min(subset)} to {np.max(subset)}")
    
    # Save subset
    tifffile.imwrite(output_path, subset, compression='lzw')
    print(f"Saved to: {output_path}")

def save_as_he5(data, output_path, wavelengths=None):
    """
    Save hyperspectral data as HE5 file in ortho surface reflectance format
    
    Args:
        data: numpy array in (rows, cols, bands) format
        output_path: Path to output HE5 file
        wavelengths: Optional array of wavelengths
    """
    print(f"Saving as HE5: {output_path}")
    print(f"Data shape: {data.shape}")
    
    # Convert to (bands, rows, cols) for HDF-EOS format
    data_transposed = np.transpose(data, (2, 0, 1))
    
    with h5py.File(output_path, 'w') as f:
        # Create HDF-EOS structure
        grids = f.create_group('HDFEOS/GRIDS/HYP/Data Fields')
        
        # Save surface reflectance data
        grids.create_dataset('surface_reflectance', data=data_transposed, 
                            compression='gzip', compression_opts=4)
        
        # Add wavelengths if available
        if wavelengths is not None:
            grids.create_dataset('wavelengths', data=wavelengths)
        
        # Add metadata
        f.attrs['Description'] = 'Hyperspectral subset image'
        f.attrs['Format'] = 'HDF-EOS'
        f.attrs['Rows'] = data.shape[0]
        f.attrs['Cols'] = data.shape[1]
        f.attrs['Bands'] = data.shape[2]
    
    print(f"Saved HE5 file: {output_path}")

if __name__ == "__main__":
    input_file = "full.tif"
    output_tif = "full_subset_r400-700_c500-800.tif"
    output_he5 = "full_subset_r400-700_c500-800.he5"
    
    # Extract from row 400-700, col 500-800
    print("Loading: full.tif")
    data = tifffile.imread(input_file)
    print(f"Original shape: {data.shape}")
    print(f"Data type: {data.dtype}")
    
    # Extract subset
    row_start, row_end = 400, 700
    col_start, col_end = 500, 800
    
    if data.ndim == 3:
        subset = data[row_start:row_end, col_start:col_end, :]
    elif data.ndim == 2:
        subset = data[row_start:row_end, col_start:col_end]
    else:
        raise ValueError(f"Unsupported number of dimensions: {data.ndim}")
    
    print(f"Subset shape: {subset.shape}")
    print(f"Subset region: rows [{row_start}:{row_end}], cols [{col_start}:{col_end}]")
    print(f"Data range: {np.min(subset)} to {np.max(subset)}")
    
    # Save as TIFF
    print("\nSaving as TIFF...")
    tifffile.imwrite(output_tif, subset, compression='lzw')
    print(f"Saved to: {output_tif}")
    
    # Save as HE5
    print("\nSaving as HE5...")
    save_as_he5(subset, output_he5)
