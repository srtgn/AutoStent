#!/bin/bash
echo "Container started."
echo "Current directory: $(pwd)"
echo "Listing src/fourc_webviewer:"
ls -F src/fourc_webviewer/

export PYTHONPATH=/app/src
export PYTHONUNBUFFERED=1

# These environment variables force VTK/PyVista to use offscreen rendering
# They are also set in the Python code, but setting them here ensures they're
# available before any imports
export PYVISTA_OFF_SCREEN=true
export VTK_DEFAULT_RENDER_WINDOW_OFFSCREEN=1

echo "Starting application directly (no xvfb)..."
exec python -m fourc_webviewer.main
