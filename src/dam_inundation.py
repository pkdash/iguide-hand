#!/usr/bin/env python3
"""
Dam Inundation Analysis

This script performs inundation analysis using HAND (Height Above Nearest Drainage)
data and hydraulic geometry calculations.

"""

import os
import sys
import argparse
import logging
from pathlib import Path
from utils import setup_taudem_path, run_taudem_command


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class DamInundationAnalysis:
    """Class for performing dam inundation analysis."""

    def __init__(self, taudem_path="$HOME/taudem", output_dir="outputs"):
        """
        Initialize the inundation analysis.

        Args:
            taudem_path: Path to TauDEM installation
            output_dir: Directory containing HAND workflow outputs
        """
        self.taudem_path = os.path.expanduser(taudem_path)
        self.output_dir = Path(output_dir)

        # Set up TauDEM PATH
        setup_taudem_path(self.taudem_path)

        # Verify required input files exist
        self.verify_inputs()

    def verify_inputs(self):
        """Verify that all required input files exist."""
        required_files = [
            "hand.tif",
            "w2.tif", 
            "catchlist.csv",
            "slp.tif",
            "stage.txt",
            "forecast.csv"
        ]

        missing_files = []
        for filename in required_files:
            filepath = self.output_dir / filename
            if not filepath.exists():
                missing_files.append(filename)

        if missing_files:
            logger.error(f"Missing required input files: {missing_files}")
            logger.error("Please run the HAND dam workflow first to generate these files.")
            sys.exit(1)

        logger.info("All required input files found.")

    def run_taudem_command(self, command, description="TauDEM command"):
        """
        Execute a TauDEM command with error handling.

        Args:
            command: Command string to execute
            description: Description for logging
        """
        run_taudem_command(command, description, cwd=self.output_dir)

    def run_hydraulic_geometry(self):
        """Run hydraulic geometry calculation."""
        logger.info("Running hydraulic geometry calculation...")

        self.run_taudem_command(
            (
                "mpiexec -n 4 catchhydrogeo "
                "-hand hand.tif "
                "-catch w2.tif "
                "-catchlist catchlist.csv "
                "-slp slp.tif "
                "-h stage.txt "
                "-table hydroprop.csv"
            ),
            "Hydraulic geometry calculation"
        )

    def run_inundation_depth(self):
        """Run inundation depth calculation."""
        logger.info("Running inundation depth calculation...")

        self.run_taudem_command(
            (
                "mpiexec -n 4 inundepth "
                "-hand hand.tif "
                "-catch w2.tif "
                "-fc forecast.csv "
                "-hp hydroprop.csv "
                "-inun inundepth.tif "
                "-depth depths.csv"
            ),
            "Inundation depth calculation"
        )

    def run_analysis(self):
        """Execute the complete inundation analysis."""
        logger.info("Starting Dam Inundation Analysis...")

        try:
            # Step 1: Hydraulic geometry calculation
            self.run_hydraulic_geometry()

            # Step 2: Inundation depth calculation
            self.run_inundation_depth()

            logger.info("Dam Inundation Analysis completed successfully!")
            logger.info(f"Results saved in: {self.output_dir}")
            logger.info("- hydroprop.csv: Hydraulic properties")
            logger.info("- inundepth.tif: Inundation depth raster")
            logger.info("- depths.csv: Inundation depth summary")

        except Exception as e:
            logger.error(f"Inundation analysis failed: {str(e)}")
            sys.exit(1)


def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(
        description="Dam Inundation Analysis - Perform inundation mapping using HAND data"
    )

    parser.add_argument(
        "--taudem-path",
        default="$HOME/taudem",
        help="Path to TauDEM installation (default: $HOME/taudem)"
    )

    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Directory containing HAND workflow outputs (default: outputs)"
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Create and run inundation analysis
    analysis = DamInundationAnalysis(
        taudem_path=args.taudem_path,
        output_dir=args.output_dir
    )

    analysis.run_analysis()


if __name__ == "__main__":
    main()
