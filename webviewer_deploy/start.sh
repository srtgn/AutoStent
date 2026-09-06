#!/bin/bash
set -euo pipefail

export PYTHONPATH=/app/src
export PYTHONUNBUFFERED=1

# These environment variables force VTK/PyVista to use offscreen rendering.
# They are also set in the Python code, but setting them here ensures they're
# available before any imports
export PYVISTA_OFF_SCREEN=true
export VTK_DEFAULT_RENDER_WINDOW_OFFSCREEN=1

if [ "${DEBUG_STARTUP:-0}" = "1" ]; then
  echo "Working directory: $(pwd)"
  echo "Contents of src/fourc_webviewer:"
  ls -F src/fourc_webviewer/
fi

# The pip build of VTK renders through GLX, so it needs a display even for
# offscreen rendering - without one it logs "bad X server connection" and the
# render window stays empty. Xvfb provides that display; if it is unavailable
# the viewer still starts, only the 3D view will not render.
if command -v xvfb-run >/dev/null 2>&1; then
  echo "Starting the 4C webviewer under Xvfb"
  exec xvfb-run -a --server-args="-screen 0 1920x1080x24" \
    python -m fourc_webviewer.main
fi

echo "Xvfb not available, starting the 4C webviewer without a display"
exec python -m fourc_webviewer.main
