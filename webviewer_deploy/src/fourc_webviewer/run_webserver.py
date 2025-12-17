"""Utility to run the webserver on a defined port."""

from fourc_webviewer.fourc_webserver import FourCWebServer
from fourc_webviewer_default_files import DEFAULT_INPUT_FILE

import os

# specify server port for the app to run on
SERVER_PORT = int(os.environ.get("PORT", 8080))


def run_webviewer(fourc_yaml_file=None):
    """Runs the webviewer by creating a dedicated webserver object, starting it
    and cleaning up afterwards."""

    print(f"Starting webviewer on port {SERVER_PORT}...", flush=True)

    # use the default input file
    if fourc_yaml_file is None:
        fourc_yaml_file = DEFAULT_INPUT_FILE
        print(f"Using default input file: {fourc_yaml_file}", flush=True)

    print("Initializing FourCWebServer...", flush=True)
    fourc_webserver = FourCWebServer(fourc_yaml_file)

    # start the server after everything is set up
    print("Starting server loop...", flush=True)
    try:
        # Configure for Railway reverse proxy:
        # - open_browser=False: Don't try to open browser in containerized env
        # - show_connection_info=False: Suppress console output
        fourc_webserver.server.start(
            port=SERVER_PORT, 
            host="0.0.0.0", 
            timeout=0,
            open_browser=False,
            show_connection_info=False
        )
        print("Server loop finished.", flush=True)
    except Exception as e:
        print(f"Server crashed with error: {e}", flush=True)
        raise

    # run cleanup
    print("Cleaning up...", flush=True)
    fourc_webserver.cleanup()
