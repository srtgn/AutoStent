"""CLI utils module."""

import argparse

import fourc_webviewer.run_webserver as webserver


def main():
    """Get the CLI arguments and start the webviewer."""
    arguments = get_arguments()
    webserver.run_webviewer(**arguments)


def get_arguments():
    """Get the CLI arguments.

    Returns:
        dict: Arguments dictionary
    """
    parser = argparse.ArgumentParser(description="4C Webviewer")
    parser.add_argument(
        "--fourc_yaml_file", type=str, help="input file path to visualize"
    )

    args = parser.parse_args()

    # return arguments as dict
    return vars(args)
