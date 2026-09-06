"""Main entry point for launching the webviewer."""

from fourc_webviewer.log_utils import configure_logging

# set up before importing anything that pulls in VTK, trame or tqdm, so that
# their output is routed and formatted from the very first record on
configure_logging()

from fourc_webviewer.cli_utils import main  # noqa: E402

if __name__ == "__main__":
    main()
