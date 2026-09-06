"""Utility to run the webserver on a defined port."""

import os

from loguru import logger

from fourc_webviewer.fourc_webserver import FourCWebServer
from fourc_webviewer_default_files import DEFAULT_INPUT_FILE

# specify server port for the app to run on
SERVER_PORT = int(os.environ.get("PORT", 8080))

# path of the endpoint used by the hosting platform to check that the app is up
HEALTH_ROUTE = "/health"


def add_health_route(server):
    """Register a health endpoint answering with a plain 200.

    The web client is served through a redirect from ``/`` to ``index.html``,
    which a platform health check reads as a failure. This endpoint answers
    directly and without touching the rendering stack, so it also stays
    responsive while a large input file is being loaded.

    Args:
        server (trame_server.core.Server): server to register the route on.
    """

    def on_server_bind(wslink_server):
        from aiohttp import web

        async def health(_request):
            return web.Response(text="ok")

        wslink_server.app.router.add_route("GET", HEALTH_ROUTE, health)
        logger.info(f"Health endpoint available on {HEALTH_ROUTE}")

    server.controller.on_server_bind.add(on_server_bind)


def run_webviewer(fourc_yaml_file=None):
    """Runs the webviewer by creating a dedicated webserver object, starting it
    and cleaning up afterwards."""

    # use the default input file
    if fourc_yaml_file is None:
        fourc_yaml_file = DEFAULT_INPUT_FILE
        logger.info(f"Using default input file: {fourc_yaml_file}")

    logger.info("Initializing FourCWebServer")
    fourc_webserver = FourCWebServer(fourc_yaml_file)
    add_health_route(fourc_webserver.server)

    # start the server after everything is set up
    logger.info(f"Starting the server loop on port {SERVER_PORT}")
    try:
        # Configure for Railway reverse proxy:
        # - open_browser=False: Don't try to open browser in containerized env
        # - show_connection_info=False: Suppress console output
        # - disable_logging=True: Reduce log noise
        fourc_webserver.server.start(
            port=SERVER_PORT,
            host="0.0.0.0",
            timeout=0,
            open_browser=False,
            show_connection_info=False,
            disable_logging=True,
        )
        logger.info("Server loop finished")
    except Exception as exc:
        logger.exception(f"Server crashed with error: {exc}")
        raise
    finally:
        # run cleanup
        logger.info("Cleaning up")
        fourc_webserver.cleanup()
