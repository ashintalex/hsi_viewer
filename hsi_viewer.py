import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, TextBox
from pathlib import Path
import sys
import os
import matplotlib
try:
    import tifffile
except ImportError:
    tifffile = None

class HSIViewer:
    """Unified viewer for .he5 and .npz hyperspectral images"""
    def __init__(self, file_path):
        self.file_path = Path(file_path)
        self.data = None
        self.wavelengths = None
        self.metadata = {}
        self.file_type = self._detect_file_type()
        self._load_data()

    def _detect_file_type(self):
        ext = self.file_path.suffix.lower()
        if ext == '.he5' or ext == '.h5' or ext == '.hdf5':
            return 'he5'
        elif ext == '.npz':
            return 'npz'
        elif ext == '.npy':
            return 'npy'
        elif ext == '.tif' or ext == '.tiff':
            return 'tif'
        else:
            raise ValueError(f"Unsupported file type: {ext}")

    def _load_data(self):
        if self.file_type == 'he5':
            self._load_he5()
        elif self.file_type == 'npz':
            self._load_npz()
        elif self.file_type == 'npy':
            self._load_npy()
        elif self.file_type == 'tif':
            self._load_tif()

    def _load_he5(self):
        """Load PRISMA or ortho surface reflectance data from HE5 file"""
        print(f"Loading: {self.file_path.name}")
        
        try:
            h5_file = h5py.File(self.file_path, 'r')
        except OSError as e:
            if "truncated file" in str(e):
                file_size_mb = self.file_path.stat().st_size / (1024 * 1024)
                raise ValueError(
                    f"HDF5 file is corrupted or incomplete.\n"
                    f"File: {self.file_path.name}\n"
                    f"Current size: {file_size_mb:.1f} MB\n"
                    f"This usually means the download or transfer was interrupted.\n"
                    f"Please re-download or re-transfer the complete file."
                ) from e
            else:
                raise
        
        with h5_file as f:
            # Print structure for debugging
            print("\nHE5 Structure:")
            self._print_structure(f)
            
            # Check if this is an ortho surface reflectance file (GRIDS format)
            if 'HDFEOS/GRIDS/HYP/Data Fields/surface_reflectance' in f:
                print("\nDetected ortho surface reflectance format")
                self._load_ortho_h5(f)
                return
            
            # Try different PRISMA data paths
            vnir_paths = [
                'HDFEOS/SWATHS/PRS_L2C_HCO/Data Fields/VNIR_Cube',  # L2C data
                'HDFEOS/SWATHS/PRS_L1_HCO/Data Fields/VNIR_Cube',
                'HDFEOS/SWATHS/PRS_L1_STD/Data Fields/VNIR_Cube', 
                'HDFEOS/SWATHS/PRS_L2D_HCO/Data Fields/VNIR_Cube',
            ]
            
            swir_paths = [
                'HDFEOS/SWATHS/PRS_L2C_HCO/Data Fields/SWIR_Cube',  # L2C data
                'HDFEOS/SWATHS/PRS_L1_HCO/Data Fields/SWIR_Cube',
                'HDFEOS/SWATHS/PRS_L1_STD/Data Fields/SWIR_Cube',
                'HDFEOS/SWATHS/PRS_L2D_HCO/Data Fields/SWIR_Cube',
            ]
            
            # Load VNIR
            self.vnir_cube = None
            for path in vnir_paths:
                if path in f:
                    self.vnir_cube = f[path][:]
                    print(f"VNIR shape: {self.vnir_cube.shape}")
                    print(f"  Data range: {np.nanmin(self.vnir_cube):.4f} to {np.nanmax(self.vnir_cube):.4f}")
                    print(f"  Non-zero/non-nan: {np.count_nonzero(~np.isnan(self.vnir_cube) & (self.vnir_cube != 0))}/{self.vnir_cube.size}")
                    break
            
            # Load SWIR
            self.swir_cube = None
            for path in swir_paths:
                if path in f:
                    self.swir_cube = f[path][:]
                    print(f"SWIR shape: {self.swir_cube.shape}")
                    break
            
            # Load wavelengths
            wvl_paths = [
                'HDFEOS/SWATHS/PRS_L1_HCO/Data Fields/VNIR_WAVELENGTHS',
                'HDFEOS/SWATHS/PRS_L1_STD/Instrument Data/VNIR_CENTER_WAVELENGTHS',
                'KDP_AUX/Cw_Vnir_Matrix',
            ]
            
            self.vnir_wavelengths = None
            for path in wvl_paths:
                if path in f:
                    wvl_data = f[path][:]
                    if wvl_data.ndim == 2:
                        self.vnir_wavelengths = np.mean(wvl_data, axis=0)
                    else:
                        self.vnir_wavelengths = wvl_data
                    print(f"VNIR wavelengths: {self.vnir_wavelengths.shape}")
                    # Remove zeros and get only valid wavelengths
                    valid_vnir_wvl = self.vnir_wavelengths[self.vnir_wavelengths > 0]
                    print(f"  Min: {np.min(self.vnir_wavelengths):.2f}, Max: {np.max(self.vnir_wavelengths):.2f}")
                    print(f"  Non-zero values: {len(valid_vnir_wvl)}/{len(self.vnir_wavelengths)}")
                    if len(valid_vnir_wvl) > 0:
                        print(f"  Valid wavelength range: {valid_vnir_wvl.min():.2f} - {valid_vnir_wvl.max():.2f} nm")
                    break
            
            # SWIR wavelengths
            swir_wvl_paths = [
                'HDFEOS/SWATHS/PRS_L1_HCO/Data Fields/SWIR_WAVELENGTHS',
                'HDFEOS/SWATHS/PRS_L1_STD/Instrument Data/SWIR_CENTER_WAVELENGTHS',
                'KDP_AUX/Cw_Swir_Matrix',
            ]
            
            self.swir_wavelengths = None
            for path in swir_wvl_paths:
                if path in f:
                    wvl_data = f[path][:]
                    if wvl_data.ndim == 2:
                        self.swir_wavelengths = np.mean(wvl_data, axis=0)
                    else:
                        self.swir_wavelengths = wvl_data
                    print(f"SWIR wavelengths: {self.swir_wavelengths.shape}")
                    # Remove zeros and get only valid wavelengths
                    valid_swir_wvl = self.swir_wavelengths[self.swir_wavelengths > 0]
                    print(f"  Min: {np.min(self.swir_wavelengths):.2f}, Max: {np.max(self.swir_wavelengths):.2f}")
                    print(f"  Non-zero values: {len(valid_swir_wvl)}/{len(self.swir_wavelengths)}")
                    if len(valid_swir_wvl) > 0:
                        print(f"  Valid wavelength range: {valid_swir_wvl.min():.2f} - {valid_swir_wvl.max():.2f} nm")
                    break
            
            # Load metadata
            self._load_metadata(f)
        
        # Combine cubes if both exist
        if self.vnir_cube is not None and self.swir_cube is not None:
            # PRISMA format: (bands, rows, cols) or (rows, bands, cols)
            # Standardize to (rows, cols, bands)
            self.vnir_cube = self._standardize_cube(self.vnir_cube)
            self.swir_cube = self._standardize_cube(self.swir_cube)
            
            print(f"Standardized VNIR: {self.vnir_cube.shape}")
            print(f"Standardized SWIR: {self.swir_cube.shape}")
            
            # Concatenate
            self.data = np.concatenate([self.vnir_cube, self.swir_cube], axis=2)
            
            if self.vnir_wavelengths is not None and self.swir_wavelengths is not None:
                # Extract only non-zero wavelengths and match to actual band count
                vnir_bands = self.vnir_cube.shape[2]
                swir_bands = self.swir_cube.shape[2]
                
                # Get non-zero wavelengths
                vnir_wvl_nonzero = self.vnir_wavelengths[self.vnir_wavelengths > 0]
                swir_wvl_nonzero = self.swir_wavelengths[self.swir_wavelengths > 0]
                
                print(f"Non-zero VNIR wavelengths: {len(vnir_wvl_nonzero)}, VNIR bands: {vnir_bands}")
                print(f"Non-zero SWIR wavelengths: {len(swir_wvl_nonzero)}, SWIR bands: {swir_bands}")
                
                # Always create proper length wavelength arrays to match band counts
                # Use valid wavelengths if available, otherwise create estimates
                if len(vnir_wvl_nonzero) >= vnir_bands:
                    vnir_wvl_final = vnir_wvl_nonzero[:vnir_bands]
                else:
                    # Not enough valid wavelengths, create placeholder
                    print(f"Creating estimated wavelengths for VNIR ({vnir_bands} bands)")
                    vnir_wvl_final = np.linspace(400, 1000, vnir_bands)
                
                if len(swir_wvl_nonzero) >= swir_bands:
                    swir_wvl_final = swir_wvl_nonzero[:swir_bands]
                else:
                    # Not enough valid wavelengths, fill the gap
                    print(f"Creating estimated wavelengths for {swir_bands - len(swir_wvl_nonzero)} missing SWIR bands")
                    # Use valid wavelengths and extend with linear interpolation
                    if len(swir_wvl_nonzero) > 0:
                        # Extend the existing wavelengths
                        last_wvl = swir_wvl_nonzero[-1]
                        gap_size = swir_bands - len(swir_wvl_nonzero)
                        extended = np.linspace(last_wvl + 10, last_wvl + 10 + gap_size * 10, gap_size)
                        swir_wvl_final = np.concatenate([swir_wvl_nonzero, extended])
                    else:
                        swir_wvl_final = np.linspace(1000, 2500, swir_bands)
                
                self.wavelengths = np.concatenate([vnir_wvl_final, swir_wvl_final])
                print(f"Final wavelength array length: {len(self.wavelengths)}, Data bands: {self.data.shape[2]}")
        elif self.vnir_cube is not None:
            self.data = self._standardize_cube(self.vnir_cube)
            if self.vnir_wavelengths is not None:
                vnir_bands = self.data.shape[2]
                vnir_wvl_nonzero = self.vnir_wavelengths[self.vnir_wavelengths > 0]
                if len(vnir_wvl_nonzero) >= vnir_bands:
                    self.wavelengths = vnir_wvl_nonzero[:vnir_bands]
                else:
                    self.wavelengths = np.linspace(400, 1000, vnir_bands)
        elif self.swir_cube is not None:
            self.data = self._standardize_cube(self.swir_cube)
            if self.swir_wavelengths is not None:
                swir_bands = self.data.shape[2]
                swir_wvl_nonzero = self.swir_wavelengths[self.swir_wavelengths > 0]
                if len(swir_wvl_nonzero) >= swir_bands:
                    self.wavelengths = swir_wvl_nonzero[:swir_bands]
                else:
                    self.wavelengths = np.linspace(1000, 2500, swir_bands)
        else:
            raise ValueError("Could not find VNIR or SWIR data cubes in HE5 file")
        
        print(f"\nFinal data cube: {self.data.shape} (rows, cols, bands)")
        if self.wavelengths is not None:
            print(f"Wavelength range: {self.wavelengths.min():.1f} - {self.wavelengths.max():.1f} nm")

    def _load_ortho_h5(self, f):
        """Load ortho surface reflectance data from GRIDS format HE5 file"""
        # Load surface reflectance data
        sr_path = 'HDFEOS/GRIDS/HYP/Data Fields/surface_reflectance'
        if sr_path in f:
            data = f[sr_path][:]
            print(f"Surface reflectance shape: {data.shape}")
            print(f"  Data range: {np.nanmin(data):.4f} to {np.nanmax(data):.4f}")
            
            # Replace fill values (typically -9999) with NaN
            data = data.astype(np.float32)
            data[data < -1000] = np.nan
            print(f"  After replacing fill values: {np.nanmin(data):.4f} to {np.nanmax(data):.4f}")
            
            # Data is in (bands, rows, cols) format - convert to (rows, cols, bands)
            self.data = np.transpose(data, (1, 2, 0))
            print(f"Standardized to: {self.data.shape} (rows, cols, bands)")
            
            # Try to load wavelength information from metadata
            # Check StructMetadata for wavelength info
            self.wavelengths = None
            if 'HDFEOS INFORMATION/StructMetadata.0' in f:
                try:
                    metadata = f['HDFEOS INFORMATION/StructMetadata.0'][()].decode('utf-8')
                    # Parse wavelength information if available in metadata
                    # For now, create estimated wavelengths based on typical hyperspectral ranges
                    n_bands = self.data.shape[2]
                    print(f"Creating estimated wavelengths for {n_bands} bands")
                    # Typical hyperspectral range: 400-2500 nm
                    self.wavelengths = np.linspace(400, 2500, n_bands)
                except Exception as e:
                    print(f"Could not parse metadata: {e}")
                    n_bands = self.data.shape[2]
                    self.wavelengths = np.linspace(400, 2500, n_bands)
            else:
                # No metadata, create estimated wavelengths
                n_bands = self.data.shape[2]
                print(f"No metadata found, creating estimated wavelengths for {n_bands} bands")
                self.wavelengths = np.linspace(400, 2500, n_bands)
            
            print(f"Wavelength range: {self.wavelengths.min():.1f} - {self.wavelengths.max():.1f} nm")
        else:
            raise ValueError("Could not find surface reflectance data in HE5 file")

    def _load_npz(self):
        arr = np.load(self.file_path)
        if 'reflectance' in arr:
            self.data = arr['reflectance']
        else:
            raise ValueError(".npz file must contain 'reflectance' array")
        self.wavelengths = None  # Could be extended if wavelength info is present
        print(f"Loaded .npz file: {self.data.shape}")

    def _load_npy(self):
        """Load hyperspectral data from .npy file"""
        data = np.load(self.file_path)
        print(f"Loaded .npy file with shape: {data.shape}")
        
        # Standardize to (rows, cols, bands) format
        self.data = self._standardize_cube(data)
        print(f"Standardized shape: {self.data.shape}")
        
        self.wavelengths = None  # No wavelength info in raw .npy files
        print(f"Loaded .npy file: {self.data.shape}")

    def _load_tif(self):
        """Load hyperspectral data from TIFF file"""
        if tifffile is None:
            raise ImportError(
                "tifffile library is required for TIFF support.\n"
                "Install it with: pip install tifffile"
            )
        
        print(f"Loading TIFF: {self.file_path.name}")
        
        # First, inspect the TIFF structure
        try:
            with tifffile.TiffFile(self.file_path) as tif:
                print(f"TIFF info:")
                print(f"  Number of pages: {len(tif.pages)}")
                if len(tif.pages) > 0:
                    page0 = tif.pages[0]
                    print(f"  Page 0 shape: {page0.shape}")
                    print(f"  Page 0 dtype: {page0.dtype}")
                    if hasattr(page0, 'tags'):
                        if 'ImageDescription' in page0.tags:
                            desc = page0.tags['ImageDescription'].value
                            if isinstance(desc, bytes):
                                desc = desc.decode('utf-8', errors='ignore')
                            print(f"  Description: {desc[:200]}")
        except Exception as e:
            print(f"Could not inspect TIFF structure: {e}")
        
        # Try to load the TIFF file
        data = None
        try:
            # Try standard loading first
            data = tifffile.imread(self.file_path)
            print(f"Loaded TIFF with shape: {data.shape}")
            print(f"  Data type: {data.dtype}")
            print(f"  Data range: {np.min(data)} to {np.max(data)}")
        except (ValueError, tifffile.TiffFileError) as e:
            print(f"Standard loading failed: {e}")
            print("Attempting page-by-page loading...")
            
            # Try loading page by page
            try:
                with tifffile.TiffFile(self.file_path) as tif:
                    pages = []
                    for i, page in enumerate(tif.pages):
                        try:
                            page_data = page.asarray()
                            pages.append(page_data)
                            if i == 0:
                                print(f"  Successfully loaded page {i}: {page_data.shape}")
                        except Exception as page_error:
                            print(f"  Failed to load page {i}: {page_error}")
                            # Try to continue with other pages
                            continue
                    
                    if len(pages) == 0:
                        raise ValueError("Could not load any pages from TIFF file")
                    elif len(pages) == 1:
                        data = pages[0]
                    else:
                        # Stack pages
                        data = np.stack(pages, axis=0)
                    
                    print(f"Loaded {len(pages)} pages, final shape: {data.shape}")
                    print(f"  Data type: {data.dtype}")
                    print(f"  Data range: {np.min(data)} to {np.max(data)}")
            except Exception as e2:
                raise ValueError(f"Could not load TIFF file: {e}\nPage-by-page loading also failed: {e2}")
        
        # Handle different TIFF formats
        if data.ndim == 2:
            # Single-band grayscale image
            print("Single-band image detected")
            # Add a third dimension for consistency
            self.data = data[:, :, np.newaxis]
        elif data.ndim == 3:
            # Could be (rows, cols, bands) or multi-page (pages, rows, cols)
            # Or RGB (rows, cols, 3)
            if data.shape[2] <= 4:  # Likely RGB/RGBA
                print(f"RGB/RGBA image detected ({data.shape[2]} channels)")
                self.data = data
            else:
                # Likely already (rows, cols, bands) for hyperspectral
                print(f"Multi-band image detected ({data.shape[2]} bands)")
                self.data = data
            
            # Check if it might be (pages, rows, cols) instead
            # Heuristic: if first dimension is small, might be pages/bands
            if data.shape[0] < data.shape[2] and data.shape[0] < 1000:
                print(f"Detected possible (bands, rows, cols) format")
                self.data = self._standardize_cube(data)
        elif data.ndim == 4:
            # Could be (pages, rows, cols, channels)
            print(f"4D TIFF detected: {data.shape}")
            if data.shape[3] <= 4:  # channels dimension
                # Reshape (pages, rows, cols, channels) to (rows, cols, pages*channels)
                print("Treating as multi-page RGB/multichannel")
                pages, rows, cols, channels = data.shape
                self.data = data.transpose(1, 2, 0, 3).reshape(rows, cols, pages * channels)
            else:
                # Unusual format, try to standardize
                self.data = data.reshape(data.shape[1], data.shape[2], -1)
        else:
            raise ValueError(f"Unsupported TIFF dimension: {data.ndim}D")
        
        print(f"Final shape: {self.data.shape} (rows, cols, bands)")
        
        # Try to read wavelength information from TIFF tags if available
        self.wavelengths = None
        try:
            with tifffile.TiffFile(self.file_path) as tif:
                # Check for wavelength metadata in tags
                first_page = tif.pages[0]
                if hasattr(first_page, 'tags'):
                    # Some hyperspectral TIFFs store wavelength info in description or custom tags
                    if 'ImageDescription' in first_page.tags:
                        desc = first_page.tags['ImageDescription'].value
                        print(f"TIFF description: {desc[:200]}...")  # Show first 200 chars
        except Exception as e:
            print(f"Could not read TIFF metadata: {e}")
        
        # If no wavelength info found, create estimated wavelengths
        if self.wavelengths is None:
            n_bands = self.data.shape[2]
            if n_bands == 1:
                self.wavelengths = np.array([550.0])  # Assume visible light
            elif n_bands == 3:
                self.wavelengths = np.array([650.0, 550.0, 450.0])  # RGB
            elif n_bands == 4:
                self.wavelengths = np.array([650.0, 550.0, 450.0, 800.0])  # RGBA or NIR
            else:
                # Hyperspectral - estimate based on typical range
                print(f"Creating estimated wavelengths for {n_bands} bands")
                self.wavelengths = np.linspace(400, 2500, n_bands)
            print(f"Wavelength range: {self.wavelengths.min():.1f} - {self.wavelengths.max():.1f} nm")

    def _standardize_cube(self, cube):
        """Standardize cube to (rows, cols, bands) format"""
        if cube.ndim != 3:
            raise ValueError(f"Expected 3D cube, got {cube.ndim}D")
        
        # PRISMA typically has (bands, rows, cols) or (rows, bands, cols)
        # We want (rows, cols, bands)
        
        # Heuristic: bands dimension is usually the SMALLEST for hyperspectral data
        # (e.g., 66 bands vs 1178x1198 pixels)
        shapes = cube.shape
        band_dim = np.argmin(shapes)
        
        if band_dim == 0:
            # (bands, rows, cols) -> (rows, cols, bands)
            return np.transpose(cube, (1, 2, 0))
        elif band_dim == 1:
            # (rows, bands, cols) -> (rows, cols, bands)
            return np.transpose(cube, (0, 2, 1))
        else:
            # Already (rows, cols, bands)
            return cube
    
    def _print_structure(self, h5_obj, prefix=''):
        """Print HDF5 structure"""
        for key in list(h5_obj.keys())[:10]:  # Limit output
            item = h5_obj[key]
            if isinstance(item, h5py.Group):
                print(f"{prefix}{key}/")
                if len(prefix) < 40:  # Limit depth
                    self._print_structure(item, prefix + '  ')
            else:
                shape = item.shape if hasattr(item, 'shape') else 'scalar'
                print(f"{prefix}{key}: {shape}")
    
    def _load_metadata(self, f):
        """Load metadata from HE5 file"""
        # Try to get acquisition info
        attr_paths = [
            'HDFEOS/SWATHS/PRS_L1_HCO',
            'HDFEOS/SWATHS/PRS_L1_STD',
        ]
        
        for path in attr_paths:
            if path in f:
                group = f[path]
                if hasattr(group, 'attrs'):
                    for key in group.attrs.keys():
                        self.metadata[key] = group.attrs[key]
    
    def find_band(self, target_wavelength):
        """Find band index closest to target wavelength"""
        if self.wavelengths is None:
            return min(int(target_wavelength), self.data.shape[2] - 1)
        return np.argmin(np.abs(self.wavelengths - target_wavelength))
    
    def get_rgb_bands(self):
        """Get band indices for RGB display (R=650nm, G=550nm, B=450nm)"""
        if self.wavelengths is not None:
            r_band = self.find_band(650)
            g_band = self.find_band(550)
            b_band = self.find_band(450)
        else:
            # For .npz files without wavelength info, use approximate band indices
            # Assuming bands are ordered by increasing wavelength
            n_bands = self.data.shape[2]
            # Approximate: R ~650nm (higher band), G ~550nm (middle), B ~450nm (lower)
            r_band = min(60, n_bands - 1)
            g_band = min(30, n_bands - 1)
            b_band = min(10, n_bands - 1)
        return r_band, g_band, b_band
    
    def get_default_single_band(self):
        """Get default band index for Single Band display (NIR ~860nm)"""
        if self.wavelengths is not None:
            return self.find_band(860)
        else:
            # For .npz files without wavelength info, use approximate band index
            n_bands = self.data.shape[2]
            # Approximate NIR band
            return min(40, n_bands - 1)
    
    def create_rgb(self, r_band=None, g_band=None, b_band=None, percentile=2):
        """Create RGB composite with percentile stretch"""
        if r_band is None:
            r_band, g_band, b_band = self.get_rgb_bands()
        
        rgb = np.stack([
            self.data[:, :, r_band],
            self.data[:, :, g_band],
            self.data[:, :, b_band]
        ], axis=2).astype(np.float32)
        
        # Percentile stretch for each channel
        for i in range(3):
            channel = rgb[:, :, i]
            valid = channel[~np.isnan(channel) & (channel > 0)]
            if len(valid) > 0:
                vmin = np.percentile(valid, percentile)
                vmax = np.percentile(valid, 100 - percentile)
                rgb[:, :, i] = np.clip((channel - vmin) / (vmax - vmin), 0, 1)
        
        return rgb
    
    def show_rgb(self, ax=None, title="RGB Composite"):
        """Display RGB composite"""
        rgb = self.create_rgb()
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 10))
        
        ax.imshow(rgb)
        ax.set_title(title)
        ax.axis('off')
        
        return ax
    
    def show_band(self, band_idx, ax=None, cmap='viridis'):
        """Display single band"""
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 10))
        
        data = self.data[:, :, band_idx]
        
        # Percentile stretch
        valid = data[~np.isnan(data) & (data > 0)]
        if len(valid) > 0:
            vmin = np.percentile(valid, 2)
            vmax = np.percentile(valid, 98)
        else:
            vmin, vmax = data.min(), data.max()
        
        im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
        
        if self.wavelengths is not None:
            wvl = self.wavelengths[band_idx]
            ax.set_title(f"Band {band_idx} ({wvl:.1f} nm)")
        else:
            ax.set_title(f"Band {band_idx}")
        
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        return ax
    
    def show_spectrum(self, row, col, ax=None):
        """Show spectral profile at pixel location"""
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 4))
        
        spectrum = self.data[row, col, :]
        
        if self.wavelengths is not None:
            ax.plot(self.wavelengths, spectrum, 'b-', linewidth=0.5)
            ax.set_xlabel('Wavelength (nm)')
        else:
            ax.plot(spectrum, 'b-', linewidth=0.5)
            ax.set_xlabel('Band Index')
        
        ax.set_ylabel('Radiance')
        # ax.set_title(f'Spectrum at pixel ({row}, {col})')
        ax.grid(True, alpha=0.3)
        
        return ax
    
    def interactive_view(self):
        """Launch interactive viewer"""
        fig = plt.figure(figsize=(16, 10))
        
        # Layout
        ax_img = fig.add_axes([0.05, 0.25, 0.55, 0.7])
        ax_spec = fig.add_axes([0.65, 0.55, 0.32, 0.35])
        ax_slider = fig.add_axes([0.1, 0.1, 0.4, 0.03])
        ax_r_slider = fig.add_axes([0.1, 0.16, 0.4, 0.02])
        ax_g_slider = fig.add_axes([0.1, 0.19, 0.4, 0.02])
        ax_b_slider = fig.add_axes([0.1, 0.22, 0.4, 0.02])
        ax_radio = fig.add_axes([0.65, 0.1, 0.15, 0.35])
        ax_info = fig.add_axes([0.82, 0.35, 0.16, 0.15])
        ax_save_btn = fig.add_axes([0.85, 0.25, 0.1, 0.05])
        ax_prev_btn = fig.add_axes([0.2, 0.05, 0.05, 0.03])
        ax_next_btn = fig.add_axes([0.3, 0.05, 0.05, 0.03])
        
        # Initial RGB display
        r_init, g_init, b_init = self.get_rgb_bands()
        rgb = self.create_rgb(r_init, g_init, b_init)
        img_display = ax_img.imshow(rgb)
        ax_img.axis('off')
        
        # Info box setup
        ax_info.axis('off')
        info_text = ax_info.text(0.5, 0.5, '', ha='center', va='center', fontsize=9, 
                                 wrap=True, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Spectrum plot - create two lines for VNIR and SWIR
        center_row, center_col = self.data.shape[0] // 2, self.data.shape[1] // 2
        vnir_line, = ax_spec.plot([], [], 'b-', linewidth=0.8, label='VNIR')
        swir_line, = ax_spec.plot([], [], 'r-', linewidth=0.8, label='SWIR')
        ax_spec.set_xlabel('Wavelength (nm)' if self.wavelengths is not None else 'Band')
        ax_spec.set_ylabel('Radiance')
        ax_spec.grid(True, alpha=0.3)
        ax_spec.legend(loc='upper right', fontsize=8)
        
        # Current state
        state = {'mode': 'RGB', 'band': self.get_default_single_band(), 'save_counter': 0, 'r_band': r_init, 'g_band': g_init, 'b_band': b_init, 'current_row': center_row, 'current_col': center_col, 'marker': None}
        
        # Band slider
        n_bands = self.data.shape[2]
        slider = Slider(ax_slider, 'Band', 0, n_bands - 1, valinit=state['band'], valstep=1)
        
        # RGB sliders
        r_slider = Slider(ax_r_slider, 'R', 0, n_bands - 1, valinit=r_init, valstep=1, color='red')
        g_slider = Slider(ax_g_slider, 'G', 0, n_bands - 1, valinit=g_init, valstep=1, color='green')
        b_slider = Slider(ax_b_slider, 'B', 0, n_bands - 1, valinit=b_init, valstep=1, color='blue')
        
        # View mode buttons
        ax_rgb_btn = fig.add_axes([0.65, 0.35, 0.15, 0.05])
        ax_sb_btn = fig.add_axes([0.65, 0.29, 0.15, 0.05])
        rgb_btn = Button(ax_rgb_btn, 'RGB', color='lightblue', hovercolor='skyblue')
        sb_btn = Button(ax_sb_btn, 'Single Band', color='lightgray', hovercolor='gray')
        
        # Save button
        save_btn = Button(ax_save_btn, 'Save Image', color='lightblue', hovercolor='skyblue')
        
        # Band navigation buttons
        prev_btn = Button(ax_prev_btn, '◄ Prev', color='lightgray', hovercolor='gray')
        next_btn = Button(ax_next_btn, 'Next ►', color='lightgray', hovercolor='gray')
        
        # Pixel coordinate input boxes
        ax_row_input = fig.add_axes([0.65, 0.91, 0.08, 0.03])
        ax_col_input = fig.add_axes([0.75, 0.91, 0.08, 0.03])
        ax_go_btn = fig.add_axes([0.85, 0.91, 0.08, 0.03])
        
        row_textbox = TextBox(ax_row_input, 'Row:', initial=str(center_row))
        col_textbox = TextBox(ax_col_input, 'Col:', initial=str(center_col))
        go_btn = Button(ax_go_btn, 'Go', color='lightgreen', hovercolor='green')
        
        def update_spectrum(row, col):
            """Update spectrum display for given pixel coordinates"""
            if 0 <= row < self.data.shape[0] and 0 <= col < self.data.shape[1]:
                spectrum = self.data[row, col, :]
                
                if self.wavelengths is not None:
                    # Split into VNIR and SWIR based on cube sizes
                    if hasattr(self, 'vnir_cube') and hasattr(self, 'swir_cube'):
                        vnir_bands = self.vnir_cube.shape[2]
                        
                        # Split spectrum and wavelengths
                        vnir_spec = spectrum[:vnir_bands]
                        swir_spec = spectrum[vnir_bands:]
                        vnir_wvl = self.wavelengths[:vnir_bands]
                        swir_wvl = self.wavelengths[vnir_bands:]
                        
                        vnir_line.set_data(vnir_wvl, vnir_spec)
                        swir_line.set_data(swir_wvl, swir_spec)
                    else:
                        # Single cube, plot as one
                        vnir_line.set_data(self.wavelengths, spectrum)
                        swir_line.set_data([], [])
                    
                    ax_spec.set_xlim(self.wavelengths.min(), self.wavelengths.max())
                else:
                    vnir_line.set_data(np.arange(len(spectrum)), spectrum)
                    swir_line.set_data([], [])
                    ax_spec.set_xlim(0, len(spectrum))
                
                valid = spectrum[~np.isnan(spectrum)]
                if len(valid) > 0:
                    ax_spec.set_ylim(valid.min() * 0.9, valid.max() * 1.1)
                
                state['current_row'] = row
                state['current_col'] = col
                
                # Update pixel marker on image
                if state['marker']:
                    for line in state['marker']:
                        line.remove()
                state['marker'] = [
                    ax_img.plot([col-10, col+10], [row, row], 'r-', linewidth=3, zorder=10)[0],
                    ax_img.plot([col, col], [row-10, row+10], 'r-', linewidth=3, zorder=10)[0]
                ]
                
                fig.canvas.draw_idle()
            else:
                print(f"Invalid coordinates: ({row}, {col}). Image size: {self.data.shape[0]}x{self.data.shape[1]}")
        
        def on_go_click(event):
            """Handle Go button click to update spectrum"""
            try:
                row = int(row_textbox.text)
                col = int(col_textbox.text)
                # Clamp values to valid range
                row = max(0, min(row, self.data.shape[0] - 1))
                col = max(0, min(col, self.data.shape[1] - 1))
                # Update text boxes with clamped values
                row_textbox.set_val(str(row))
                col_textbox.set_val(str(col))
                update_spectrum(row, col)
            except ValueError:
                print("Please enter valid integer coordinates")
        
        def update_spectrum(row, col):
            """Update spectrum display for given pixel coordinates"""
            if 0 <= row < self.data.shape[0] and 0 <= col < self.data.shape[1]:
                spectrum = self.data[row, col, :]
                
                if self.wavelengths is not None:
                    # Split into VNIR and SWIR based on cube sizes
                    if hasattr(self, 'vnir_cube') and hasattr(self, 'swir_cube'):
                        vnir_bands = self.vnir_cube.shape[2]
                        
                        # Split spectrum and wavelengths
                        vnir_spec = spectrum[:vnir_bands]
                        swir_spec = spectrum[vnir_bands:]
                        vnir_wvl = self.wavelengths[:vnir_bands]
                        swir_wvl = self.wavelengths[vnir_bands:]
                        
                        vnir_line.set_data(vnir_wvl, vnir_spec)
                        swir_line.set_data(swir_wvl, swir_spec)
                    else:
                        # Single cube, plot as one
                        vnir_line.set_data(self.wavelengths, spectrum)
                        swir_line.set_data([], [])
                    
                    ax_spec.set_xlim(self.wavelengths.min(), self.wavelengths.max())
                else:
                    vnir_line.set_data(np.arange(len(spectrum)), spectrum)
                    swir_line.set_data([], [])
                    ax_spec.set_xlim(0, len(spectrum))
                
                valid = spectrum[~np.isnan(spectrum)]
                if len(valid) > 0:
                    ax_spec.set_ylim(valid.min() * 0.9, valid.max() * 1.1)
                
                # ax_spec.set_title(f'Spectrum at pixel ({row}, {col})')
                state['current_row'] = row
                state['current_col'] = col
                fig.canvas.draw_idle()
            else:
                print(f"Invalid coordinates: ({row}, {col}). Image size: {self.data.shape[0]}x{self.data.shape[1]}")
        
        def update_display():
            if state['mode'] == 'RGB':
                img_display.set_data(self.create_rgb(state['r_band'], state['g_band'], state['b_band']))
                info_text.set_text("RGB Composite")
            elif state['mode'] == 'Single Band':
                band_data = self.data[:, :, state['band']]
                valid = band_data[~np.isnan(band_data) & (band_data > 0)]
                if len(valid) > 0:
                    vmin, vmax = np.percentile(valid, [2, 98])
                else:
                    vmin, vmax = 0, 1
                normalized = np.clip((band_data - vmin) / (vmax - vmin + 1e-10), 0, 1)
                img_display.set_data(plt.cm.viridis(normalized))
                info_text.set_text("Single Band")
            
            fig.canvas.draw_idle()
        
        def on_slider_change(val):
            state['band'] = int(val)
            if state['mode'] == 'Single Band':
                update_display()
        
        def on_r_slider_change(val):
            state['r_band'] = int(val)
            if state['mode'] == 'RGB':
                update_display()
        
        def on_g_slider_change(val):
            state['g_band'] = int(val)
            if state['mode'] == 'RGB':
                update_display()
        
        def on_b_slider_change(val):
            state['b_band'] = int(val)
            if state['mode'] == 'RGB':
                update_display()
        
        def on_rgb_click(event):
            state['mode'] = 'RGB'
            rgb_btn.color = 'lightblue'
            sb_btn.color = 'lightgray'
            update_display()
        
        def on_sb_click(event):
            state['mode'] = 'Single Band'
            sb_btn.color = 'lightblue'
            rgb_btn.color = 'lightgray'
            update_display()
        
        def on_save(event):
            """Save current view to file"""
            state['save_counter'] += 1
            mode_name = state['mode'].replace(' ', '_').lower()
            
            # Create data folder if it doesn't exist
            data_dir = self.file_path.parent / 'data'
            data_dir.mkdir(exist_ok=True)
            
            if state['mode'] == 'Single Band':
                filename = data_dir / f"{self.file_path.stem}_{mode_name}_band{state['band']}_{state['save_counter']}.png"
            else:
                filename = data_dir / f"{self.file_path.stem}_{mode_name}_{state['save_counter']}.png"
            
            # Save just the main image
            extent = ax_img.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
            fig.savefig(filename, bbox_inches=extent.expanded(1.1, 1.1), dpi=150)
            print(f"Saved: {filename}")
            ax_save_btn.set_title(f"Saved: {filename.name[:20]}...", fontsize=8)
            fig.canvas.draw_idle()
        
        def on_prev_band(event):
            """Go to previous band"""
            if state['band'] > 0:
                state['band'] -= 1
                slider.set_val(state['band'])
        
        def on_next_band(event):
            """Go to next band"""
            if state['band'] < n_bands - 1:
                state['band'] += 1
                slider.set_val(state['band'])
        
        def on_go_click(event):
            """Handle Go button click to update spectrum for entered coordinates"""
            try:
                row = int(row_textbox.text)
                col = int(col_textbox.text)
                
                # Clamp values to valid range
                row = max(0, min(row, self.data.shape[0] - 1))
                col = max(0, min(col, self.data.shape[1] - 1))
                
                # Update text boxes with clamped values
                row_textbox.set_val(str(row))
                col_textbox.set_val(str(col))
                
                update_spectrum(row, col)
            except ValueError:
                print("Invalid input: Please enter integer values for row and column")
        
        def on_click(event):
            if event.inaxes == ax_img:
                col, row = int(event.xdata), int(event.ydata)
                if 0 <= row < self.data.shape[0] and 0 <= col < self.data.shape[1]:
                    update_spectrum(row, col)
                    # Update text boxes with clicked coordinates
                    row_textbox.set_val(str(row))
                    col_textbox.set_val(str(col))
        
        slider.on_changed(on_slider_change)
        r_slider.on_changed(on_r_slider_change)
        g_slider.on_changed(on_g_slider_change)
        b_slider.on_changed(on_b_slider_change)
        rgb_btn.on_clicked(on_rgb_click)
        sb_btn.on_clicked(on_sb_click)
        save_btn.on_clicked(on_save)
        prev_btn.on_clicked(on_prev_band)
        next_btn.on_clicked(on_next_band)
        go_btn.on_clicked(on_go_click)
        fig.canvas.mpl_connect('button_press_event', on_click)
        
        # Initial spectrum
        update_spectrum(center_row, center_col)
        
        # Set initial info text
        info_text.set_text("RGB Composite")
        
        plt.suptitle(f"HSI Viewer: {self.file_path.name}", fontsize=12, y=0.98)
        
        # Check if we can display or need to save
        if matplotlib.get_backend() == 'Agg':
            output_file = self.file_path.stem + '_interactive_view.png'
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            print(f"\nCannot display interactive window (no display available).")
            print(f"Saved static view to: {output_file}")
        else:
            print("\nDisplaying interactive viewer window...")
            print("Click on the image to view pixel spectra")
            plt.show(block=True)
    
    def quick_view(self):
        """Quick static view with RGB, single band, and stats"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # RGB
        self.show_rgb(axes[0, 0], "RGB Composite (R=650, G=550, B=450 nm)")
        
        # Single band (NIR)
        nir_band = self.find_band(860)
        self.show_band(nir_band, axes[0, 1], 'gray')
        
        # NDVI
        nir = self.data[:, :, self.find_band(860)].astype(np.float32)
        red = self.data[:, :, self.find_band(650)].astype(np.float32)
        ndvi = (nir - red) / (nir + red + 1e-10)
        
        im = axes[1, 0].imshow(ndvi, cmap='RdYlGn', vmin=-1, vmax=1)
        axes[1, 0].set_title('NDVI')
        axes[1, 0].axis('off')
        plt.colorbar(im, ax=axes[1, 0], fraction=0.046, pad=0.04)
        
        # Mean spectrum
        mean_spectrum = np.nanmean(self.data, axis=(0, 1))
        if self.wavelengths is not None:
            axes[1, 1].plot(self.wavelengths, mean_spectrum, 'b-', linewidth=0.8)
            axes[1, 1].set_xlabel('Wavelength (nm)')
        else:
            axes[1, 1].plot(mean_spectrum, 'b-', linewidth=0.8)
            axes[1, 1].set_xlabel('Band Index')
        axes[1, 1].set_ylabel('Mean Radiance')
        axes[1, 1].set_title('Scene Mean Spectrum')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle(f"{self.file_path.name}\nShape: {self.data.shape}", fontsize=12)
        plt.tight_layout()
        
        # Check if we can display or need to save
        if matplotlib.get_backend() == 'Agg':
            output_file = self.file_path.stem + '_quick_view.png'
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            print(f"\nSaved quick view to: {output_file}")
        else:
            print("\nDisplaying quick view window...")
            plt.show(block=True)
