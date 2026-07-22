"""
utils.py

General utility functions for the prostate cancer
machine learning package.
"""

from pathlib import Path
from datetime import datetime


class Utils:
    """
    General helper functions.
    """

    @staticmethod
    def create_directory(path):
        """
        Create a directory if it does not already exist.

        Parameters
        ----------
        path : str or Path

        Returns
        -------
        Path
            Path object corresponding to the directory.
        """

        path = Path(path)

        path.mkdir(parents=True, exist_ok=True)

        return path

    @staticmethod
    def timestamp():
        """
        Return the current timestamp.

        Returns
        -------
        str
            Timestamp formatted as YYYYMMDD_HHMMSS.
        """

        return datetime.now().strftime("%Y%m%d_%H%M%S")

    @staticmethod
    def save_figure(plt, filename, dpi=300):
        """
        Save a matplotlib figure.

        Parameters
        ----------
        plt : matplotlib.pyplot

        filename : str

        dpi : int
            Figure resolution.
        """

        plt.savefig(

            filename,

            dpi=dpi,

            bbox_inches="tight"

        )

    @staticmethod
    def print_header(title):
        """
        Print a formatted console header.

        Parameters
        ----------
        title : str
        """

        print("\n" + "=" * 60)

        print(title)

        print("=" * 60)

    @staticmethod
    def print_success(message):
        """
        Print a success message.

        Parameters
        ----------
        message : str
        """

        print(f"✓ {message}")
