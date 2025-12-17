#!/bin/bash
echo "Container started."
echo "Current directory: $(pwd)"
echo "Listing src/fourc_webviewer:"
ls -F src/fourc_webviewer/

export PYTHONPATH=/app/src
export PYTHONUNBUFFERED=1

echo "Starting application with xvfb-run..."
exec xvfb-run -a --server-args="-screen 0 1024x768x24" python -m fourc_webviewer.main
