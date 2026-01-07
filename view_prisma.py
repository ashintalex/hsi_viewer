#!/usr/bin/env python3
"""
PRISMA HE5 Hyperspectral Image Viewer

Interactive viewer for PRISMA L1/L2 .he5 files with:
- RGB composite display
- Single band viewer with wavelength selection
- False color composites
- Spectral profile extraction
- Basic statistics

Usage:
    python view_prisma.py <path_to_he5_file>
    python view_prisma.py  # Opens file dialog
"""

import sys
import numpy as np
import h5py

# Set matplotlib backend for macOS before importing pyplot
import matplotlib
# Try to use TkAgg, fall back to Agg if no display available
import os

# For macOS, try different backends in order of preference
if sys.platform == 'darwin':
    # Try MacOSX backend first (native), then TkAgg, then Agg as fallback
    for backend in ['MacOSX', 'TkAgg', 'Qt5Agg', 'Agg']:
        try:
            matplotlib.use(backend, force=True)
            print(f"Using matplotlib backend: {backend}")
            break
        except:
            continue
else:
    if os.environ.get('DISPLAY') is None:
        matplotlib.use('Agg')  # Use non-interactive backend if no display
    else:
        matplotlib.use('TkAgg')  # Use TkAgg backend for better compatibility
    
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from pathlib import Path

# Try to import tkinter for file dialog
try:
    import tkinter as tk
    from tkinter import filedialog
    HAS_TK = True
except ImportError:
    HAS_TK = False


from hsi_viewer import HSIViewer


def select_file():
    """Open file dialog to select HE5 or NPZ file"""
    if not HAS_TK:
        print("tkinter not available. Please provide file path as argument.")
        sys.exit(1)
    
    root = tk.Tk()
    root.withdraw()
    
    file_path = filedialog.askopenfilename(
        title="Select PRISMA HE5 or NPZ file",
        filetypes=[("HE5 files", "*.he5"), ("HDF5 files", "*.h5 *.hdf5"), ("NPZ files", "*.npz"), ("All files", "*.*")]
    )
    
    root.destroy()
    return file_path


def main():
    """Main entry point"""
    import argparse
    parser = argparse.ArgumentParser(description='HSI Viewer for PRISMA .he5 and .npz files')
    parser.add_argument('file', nargs='?', help='Path to .he5 or .npz file')
    parser.add_argument('--quick', '-q', action='store_true', help='Quick static view instead of interactive')
    args = parser.parse_args()

    # Get file path
    if args.file:
        file_path = args.file
    else:
        file_path = select_file()
        if not file_path:
            print("No file selected")
            sys.exit(1)

    # Check file exists
    if not Path(file_path).exists():
        print(f"Error: File not found: {file_path}")
        sys.exit(1)

    # Create viewer
    try:
        viewer = HSIViewer(file_path)
        if args.quick:
            viewer.quick_view()
        else:
            viewer.interactive_view()
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
