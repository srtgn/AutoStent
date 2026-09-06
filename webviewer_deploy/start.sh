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
# render window stays empty.
#
# Xvfb is started here rather than through xvfb-run: xvfb-run waits for the X
# server to signal readiness, and that handshake never completes when it runs
# as PID 1 of a container, so it hangs before ever launching the command - the
# viewer then produces no output at all and no health check can pass.
DISPLAY_NUM="${DISPLAY_NUM:-99}"
if command -v Xvfb >/dev/null 2>&1; then
  Xvfb ":${DISPLAY_NUM}" -screen 0 1920x1080x24 -nolisten tcp &
  for _ in $(seq 1 100); do
    [ -e "/tmp/.X11-unix/X${DISPLAY_NUM}" ] && break
    sleep 0.1
  done
  if [ -e "/tmp/.X11-unix/X${DISPLAY_NUM}" ]; then
    export DISPLAY=":${DISPLAY_NUM}"
    echo "Xvfb ready on display ${DISPLAY}"
  else
    echo "Xvfb did not come up, starting the viewer without a display"
  fi
else
  echo "Xvfb not installed, starting the viewer without a display"
fi

exec python -m fourc_webviewer.main
