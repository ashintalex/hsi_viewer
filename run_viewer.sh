#!/bin/bash
# PRISMA Viewer Launcher Script

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Activate virtual environment and run the viewer
cd "$SCRIPT_DIR"
source venv/bin/activate
python view_prisma.py "$@"
