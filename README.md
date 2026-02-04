# HSI Viewer - Hyperspectral Image Viewer for PRISMA Data

A Python-based interactive viewer for PRISMA hyperspectral satellite imagery stored in HE5 or NPZ format. This tool allows you to explore and visualize hyperspectral data cubes with an intuitive graphical interface.

## Overview

PRISMA is an Italian hyperspectral satellite mission that captures images across hundreds of spectral bands. This viewer helps you work with PRISMA Level 1 and Level 2 data products by providing real-time visualization, spectral analysis, and export capabilities.

The viewer supports both VNIR (Visible and Near-Infrared) and SWIR (Short-Wave Infrared) spectral regions, displaying them as separate colored traces to clearly distinguish between the two sensor systems. It also supports .npz files containing reflectance data.

## Features

### Interactive Visualization
- Click anywhere on the image to view the full spectral signature at that pixel
- Red cross indicator shows the currently selected pixel on the image
- Real-time RGB composite creation with adjustable band selection
- Single band viewer for examining individual spectral channels
- Dual-colored spectral plots (blue for VNIR, red for SWIR)
- Manual pixel coordinate input via text boxes

### Navigation Controls
- RGB sliders for custom false-color composites
- Band slider with Previous/Next buttons for easy navigation
- Toggle buttons to switch between RGB and Single Band modes
- Text input boxes for direct pixel coordinate entry

### Data Export
- Save current view as high-resolution PNG images
- All exports automatically saved to a data folder
- Configurable output resolution

### Supported Data Products
- PRISMA L1 (HCO, STD)
- PRISMA L2C (HCO)
- PRISMA L2D (HCO)
- Ortho surface reflectance HDF5 files (GRIDS format)
- .npz files with 'reflectance' arrays
- Automatic detection of data cube locations within HE5 files

## Requirements

- Python 3.8 or higher
- numpy
- matplotlib
- h5py

## Installation

Clone this repository:

```bash
git clone https://github.com/ashintalex/hsi_viewer.git
cd hsi_viewer
```

Create a virtual environment and install dependencies:

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install numpy matplotlib h5py
```

Make the launcher script executable:

```bash
chmod +x run_viewer.sh
```

## Usage

### Quick Start

Run the viewer with a PRISMA HE5 or NPZ file:

```bash
./run_viewer.sh path/to/your/prisma_file.he5
```

Or using Python directly:

```bash
venv/bin/python view_prisma.py path/to/your/prisma_file.he5
```

For NPZ files:

```bash
venv/bin/python view_prisma.py path/to/your/data.npz
```

### Interactive Mode

The default mode opens an interactive window where you can:

1. **View RGB Composites**: Use the R, G, and B sliders to select which spectral bands to display as red, green, and blue channels
2. **Examine Single Bands**: Click the Single Band button and use the band slider or Previous/Next buttons to browse through individual spectral channels
3. **Extract Spectral Profiles**: Click any pixel in the image to see its spectral signature in the graph panel, or enter coordinates directly in the Row/Col text boxes
4. **Visual Feedback**: A red cross on the image indicates the currently selected pixel
5. **Save Images**: Click the Save Image button to export the current view to the data folder

### Quick View Mode

For a static summary view:

```bash
./run_viewer.sh your_file.he5 --quick
```

This generates a PNG file with RGB composite, single band view, NDVI, and mean spectrum.

## Understanding the Spectral Plot

The spectral graph displays radiance values across wavelengths with two colored lines:

- **Blue line**: VNIR region (approximately 400-1000 nm)
- **Red line**: SWIR region (approximately 950-2500 nm)

Note that for some L2C products, VNIR wavelengths may be estimated if calibration data is not available in the file.

## File Structure

```
hsi_viewer/
├── view_prisma.py          # Main viewer application
├── hsi_viewer.py           # HSIViewer class implementation
├── run_viewer.sh           # Launcher script
├── venv/                   # Virtual environment
├── data/                   # Output folder for saved images
└── README.md              # This file
```

## Known Limitations

- L2C data products may have limited or estimated VNIR wavelength calibration
- .npz files without wavelength information use default band indices for RGB composites
- Very large data cubes may take time to load
- Interactive mode requires a display (GUI environment)

## Troubleshooting

### "No module named h5py" error
Make sure you are using the virtual environment Python:
```bash
./run_viewer.sh your_file.he5
```
or
```bash
venv/bin/python view_prisma.py your_file.he5
```

### Window doesn't display on macOS
The viewer automatically detects the best display backend. If you encounter issues, ensure you have the latest version of matplotlib installed.

### Spectral plot shows discontinuities
This is expected for data products where VNIR and SWIR wavelengths overlap or have gaps. The two-color display helps visualize this clearly.

### .npz files not loading
Ensure the .npz file contains a 'reflectance' array with shape (rows, cols, bands). The viewer expects 3D arrays in this format.

## Contributing

Contributions are welcome. Please feel free to submit issues or pull requests for bug fixes, improvements, or new features.

## Acknowledgments

This tool was developed to support analysis of PRISMA hyperspectral satellite data from the Italian Space Agency (ASI). PRISMA data products are available through the ASI portal.

## License

MIT License - feel free to use and modify this code for your research and applications.

## Contact

For questions, issues, or suggestions, please open an issue on GitHub.
