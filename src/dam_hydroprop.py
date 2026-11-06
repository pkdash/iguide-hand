#!/usr/bin/env python3
"""
Dam Hydraulic Properties (Synthetic Rating Curve)

Runs TauDEM ``catchhydrogeo`` on the HAND outputs of ``dam_hand.py`` to build
``hydroprop.csv``: for every reach in ``catchlist.csv`` and every stage height
in ``stage.txt`` it tabulates wetted area, volume, cross-sectional area, wetted
perimeter, hydraulic radius and Manning discharge (the synthetic rating curve).

The table depends only on terrain (HAND, slope, catchments) and reach
hydraulics (slope, length, Manning's n), so it is computed once per HAND run
and reused by ``dam_inundation.py`` for any number of flow scenarios.
"""

import os
import sys
import shlex
import argparse
import logging
from pathlib import Path

import pandas as pd

from utils import setup_taudem_path, run_taudem_command


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Fewer stages than this gives a coarse rating curve to interpolate on
MIN_RECOMMENDED_STAGES = 5


def load_stage_heights(stage_file):
    """
    Read and validate stage heights from ``stage.txt``.

    The file holds one stage height (m) per line, optionally preceded by a
    non-numeric header line (e.g. ``Stage``).

    Returns:
        list[float]: stage heights in file order

    Raises:
        ValueError: if a value is non-numeric or non-positive, or if the
            stages are not strictly increasing
    """
    lines = [ln.strip() for ln in Path(stage_file).read_text().splitlines()]
    lines = [ln for ln in lines if ln]
    if not lines:
        raise ValueError(f"Stage file is empty: {stage_file}")

    # Skip a header line if the first entry is not a number
    try:
        float(lines[0])
    except ValueError:
        lines = lines[1:]

    stages = []
    for i, ln in enumerate(lines, start=1):
        try:
            stages.append(float(ln))
        except ValueError:
            raise ValueError(f"Non-numeric stage value '{ln}' (data line {i}) in {stage_file}")

    if not stages:
        raise ValueError(f"No stage values found in {stage_file}")

    non_positive = [s for s in stages if s <= 0]
    if non_positive:
        raise ValueError(f"Stage heights must be > 0; found {non_positive} in {stage_file}")

    for prev, cur in zip(stages, stages[1:]):
        if cur <= prev:
            raise ValueError(
                f"Stage heights must be strictly increasing; found {prev} followed by {cur} "
                f"in {stage_file}"
            )

    return stages


class DamHydroProp:
    """Compute reach hydraulic properties (synthetic rating curves) from HAND outputs."""

    def __init__(
        self,
        taudem_path="$HOME/taudem",
        output_dir="outputs",
        hand_raster=None,
        catch_raster=None,
        catchlist=None,
        slope_raster=None,
        stage_file=None,
        hydroprop_file=None,
        np=4,
    ):
        """
        Initialize the hydraulic properties computation.

        Args:
            taudem_path: Path to TauDEM installation
            output_dir: Directory containing HAND workflow outputs; also the
                working directory and the default location of every file below
            hand_raster: HAND raster (default: <output_dir>/hand.tif)
            catch_raster: Reach catchment raster (default: <output_dir>/w2.tif)
            catchlist: Reach list CSV (default: <output_dir>/catchlist.csv)
            slope_raster: Slope raster (default: <output_dir>/slp.tif)
            stage_file: Stage heights file (default: <output_dir>/stage.txt)
            hydroprop_file: Output table (default: <output_dir>/hydroprop.csv)
            np: Number of MPI processes
        """
        self.taudem_path = Path(os.path.expandvars(taudem_path)).expanduser().resolve()
        self.output_dir = Path(output_dir).resolve()

        def _resolve(path, default_name):
            return Path(path).resolve() if path else self.output_dir / default_name

        self.hand_raster = _resolve(hand_raster, "hand.tif")
        self.catch_raster = _resolve(catch_raster, "w2.tif")
        self.catchlist = _resolve(catchlist, "catchlist.csv")
        self.slope_raster = _resolve(slope_raster, "slp.tif")
        self.stage_file = _resolve(stage_file, "stage.txt")
        self.hydroprop_file = _resolve(hydroprop_file, "hydroprop.csv")
        self.np = np

        # Set up TauDEM PATH
        setup_taudem_path(str(self.taudem_path))

        # Verify required input files exist
        self.verify_inputs()

    def verify_inputs(self):
        """Verify that all required input files exist."""
        required_files = [
            self.hand_raster,
            self.catch_raster,
            self.catchlist,
            self.slope_raster,
            self.stage_file,
        ]

        missing_files = [str(f) for f in required_files if not f.exists()]
        if missing_files:
            logger.error(f"Missing required input files: {missing_files}")
            logger.error("Please run the HAND dam workflow (dam_hand) first to generate these files.")
            sys.exit(1)

        logger.info("All required input files found.")

    def validate_stage_file(self):
        """Validate stage heights; exit on invalid input."""
        try:
            stages = load_stage_heights(self.stage_file)
        except ValueError as e:
            logger.error(str(e))
            sys.exit(1)

        if len(stages) < MIN_RECOMMENDED_STAGES:
            logger.warning(
                f"Only {len(stages)} stage heights in {self.stage_file}; the rating curve "
                f"will be coarse. Consider at least {MIN_RECOMMENDED_STAGES}."
            )
        logger.info(
            f"Stage heights: {len(stages)} values from {stages[0]} m to {stages[-1]} m"
        )
        return stages

    def run_hydraulic_geometry(self):
        """Run TauDEM catchhydrogeo to produce the hydraulic properties table."""
        logger.info("Running hydraulic geometry calculation...")
        q = shlex.quote
        run_taudem_command(
            (
                f"mpiexec -n {int(self.np)} catchhydrogeo "
                f"-hand {q(str(self.hand_raster))} "
                f"-catch {q(str(self.catch_raster))} "
                f"-catchlist {q(str(self.catchlist))} "
                f"-slp {q(str(self.slope_raster))} "
                f"-h {q(str(self.stage_file))} "
                f"-table {q(str(self.hydroprop_file))}"
            ),
            "Hydraulic geometry calculation",
            cwd=str(self.output_dir),
        )

    def summarize_hydroprop(self):
        """Log the rating-curve range per reach and flag reaches missing from the table."""
        if not self.hydroprop_file.exists():
            logger.error(f"catchhydrogeo did not produce {self.hydroprop_file}")
            sys.exit(1)

        hp = pd.read_csv(self.hydroprop_file, skipinitialspace=True)
        hp.columns = hp.columns.str.strip()
        needed = {"Id", "Stage_m", "Flow_m3s"}
        if not needed.issubset(hp.columns):
            logger.warning(
                f"Unexpected hydroprop.csv columns {list(hp.columns)}; skipping summary"
            )
            return

        summary = hp.groupby("Id").agg(max_stage_m=("Stage_m", "max"), max_flow_m3s=("Flow_m3s", "max"))
        logger.info("Rating-curve range per reach (largest mappable discharge):")
        for reach_id, row in summary.iterrows():
            logger.info(
                f"  Reach {reach_id}: max stage {row.max_stage_m:g} m -> Q {row.max_flow_m3s:,.1f} m3/s"
            )

        catch = pd.read_csv(self.catchlist, skipinitialspace=True)
        catch.columns = catch.columns.str.strip()
        if "Id" in catch.columns:
            missing = sorted(set(catch["Id"]) - set(summary.index))
            if missing:
                logger.warning(f"Reaches in catchlist.csv missing from hydroprop.csv: {missing}")

    def run(self):
        """Execute the hydraulic properties computation."""
        logger.info("Starting Dam Hydraulic Properties computation...")

        try:
            self.validate_stage_file()
            self.run_hydraulic_geometry()
            self.summarize_hydroprop()

            logger.info("Hydraulic properties computed successfully!")
            logger.info(f"- Hydraulic properties table: {self.hydroprop_file}")
            logger.info("Next: run dam_inundation with a forecast file.")

        except Exception as e:
            logger.error(f"Hydraulic properties computation failed: {str(e)}")
            sys.exit(1)


def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(
        description=(
            "Dam Hydraulic Properties - build the synthetic rating curve table "
            "(hydroprop.csv) from HAND outputs using TauDEM catchhydrogeo"
        )
    )

    parser.add_argument(
        "--taudem-path",
        default="$HOME/taudem",
        help="Path to TauDEM installation (default: $HOME/taudem)"
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Directory containing HAND workflow outputs; default location of all files below (default: outputs)"
    )
    parser.add_argument("--hand-raster", help="HAND raster (default: <output-dir>/hand.tif)")
    parser.add_argument("--catch-raster", help="Reach catchment raster (default: <output-dir>/w2.tif)")
    parser.add_argument("--catchlist", help="Reach list CSV (default: <output-dir>/catchlist.csv)")
    parser.add_argument("--slope-raster", help="Slope raster (default: <output-dir>/slp.tif)")
    parser.add_argument("--stage-file", help="Stage heights file, one value (m) per line (default: <output-dir>/stage.txt)")
    parser.add_argument("--hydroprop-file", help="Output hydraulic properties table (default: <output-dir>/hydroprop.csv)")
    parser.add_argument("--np", type=int, default=4, help="Number of MPI processes (default: 4)")
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    DamHydroProp(
        taudem_path=args.taudem_path,
        output_dir=args.output_dir,
        hand_raster=args.hand_raster,
        catch_raster=args.catch_raster,
        catchlist=args.catchlist,
        slope_raster=args.slope_raster,
        stage_file=args.stage_file,
        hydroprop_file=args.hydroprop_file,
        np=args.np,
    ).run()


if __name__ == "__main__":
    main()
