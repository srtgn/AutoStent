"""Utility to run the webserver on a defined port."""

from fourc_webviewer.fourc_webserver import FourCWebServer
from fourc_webviewer_default_files import DEFAULT_INPUT_FILE

import os

# specify server port for the app to run on
SERVER_PORT = int(os.environ.get("PORT", 8080))


def run_webviewer(fourc_yaml_file=None):
    """Runs the webviewer by creating a dedicated webserver object, starting it
    and cleaning up afterwards."""

    # use the default input file
    if fourc_yaml_file is None:
        fourc_yaml_file = DEFAULT_INPUT_FILE

    fourc_webserver = FourCWebServer(fourc_yaml_file)

    # start the server after everything is set up
    fourc_webserver.server.start(port=SERVER_PORT, host="0.0.0.0", timeout=0)

    # run cleanup
    fourc_webserver.cleanup()
