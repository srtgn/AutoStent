#!/bin/bash
echo "Container started."
echo "Current directory: $(pwd)"
echo "Listing src/fourc_webviewer:"
ls -F src/fourc_webviewer/

export PYTHONPATH=/app/src
export PYTHONUNBUFFERED=1

echo "Checking for xvfb-run..."
which xvfb-run
if [ $? -ne 0 ]; then
    echo "ERROR: xvfb-run not found!"
    exit 1
fi

echo "Starting application with xvfb-run..."
# Not using exec immediately to ensure we can catch exit codes if needed, 
# but exec is better for signal handling. 
# We'll just exec it now that we know it exists.
exec xvfb-run -a --server-args="-screen 0 1024x768x24" python -m fourc_webviewer.main
