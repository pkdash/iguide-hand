#!/usr/bin/env python3
"""
Dam Inundation Analysis

Maps inundation depth for a flow scenario using TauDEM ``inundepth``. For each
reach, the forecast discharge is converted to a stage via the synthetic rating
curve in ``hydroprop.csv`` (produced by ``dam_hydroprop.py``), and depth is
computed as stage minus HAND within the reach catchment.

Forecast file contract (``forecast.csv``)
-----------------------------------------
CSV with a header row and one row per reach:

- ``Id``       -- integer reach ID; must match ``Id`` in ``catchlist.csv``
                  (``LINKNO`` in ``net2.shp``) from the *same* ``dam_hand`` run.
                  Reach IDs change whenever the reach outlets change.
- ``flow_m3s`` -- discharge in m^3/s (> 0).

Whitespace after commas is allowed; extra columns are ignored. Scripts that
generate forecasts can import ``load_forecast``, ``validate_forecast_ids`` and
``validate_rating_curve_range`` from this module to check their output.
"""

import os
import sys
import shlex
import argparse
import logging
from pathlib import Path

import pandas as pd

from utils import setup_taudem_path, run_taudem_command, default_output_dir


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Forecast file schema
FORECAST_ID_COL = "Id"
FORECAST_FLOW_COL = "flow_m3s"

# hydroprop.csv columns used here (written by TauDEM catchhydrogeo)
HYDROPROP_ID_COL = "Id"
HYDROPROP_STAGE_COL = "Stage_m"
HYDROPROP_FLOW_COL = "Flow_m3s"


def _read_csv(path):
    """Read a CSV tolerating whitespace after commas; strip column names."""
    df = pd.read_csv(path, skipinitialspace=True)
    df.columns = df.columns.str.strip()
    return df


def load_forecast(path):
    """
    Read and validate a forecast file against the forecast contract.

    Returns:
        pandas.DataFrame with integer ``Id`` and float ``flow_m3s`` columns
        (extra columns are kept but ignored downstream)

    Raises:
        ValueError: if required columns are missing, IDs are non-integer or
            duplicated, or flows are missing/non-numeric
    """
    df = _read_csv(path)

    missing = [c for c in (FORECAST_ID_COL, FORECAST_FLOW_COL) if c not in df.columns]
    if missing:
        raise ValueError(
            f"Forecast file {path} is missing required column(s) {missing}; "
            f"found {list(df.columns)}. Expected '{FORECAST_ID_COL}, {FORECAST_FLOW_COL}'."
        )

    extra = [c for c in df.columns if c not in (FORECAST_ID_COL, FORECAST_FLOW_COL)]
    if extra:
        logger.warning(f"Forecast file has extra columns that will be ignored: {extra}")

    ids = pd.to_numeric(df[FORECAST_ID_COL], errors="coerce")
    if ids.isna().any() or (ids % 1 != 0).any():
        bad = df.loc[ids.isna() | (ids % 1 != 0), FORECAST_ID_COL].tolist()
        raise ValueError(f"Forecast file {path} has non-integer reach IDs: {bad}")
    df[FORECAST_ID_COL] = ids.astype(int)

    dupes = df.loc[df[FORECAST_ID_COL].duplicated(), FORECAST_ID_COL].unique().tolist()
    if dupes:
        raise ValueError(f"Forecast file {path} has duplicate reach IDs: {dupes}")

    flows = pd.to_numeric(df[FORECAST_FLOW_COL], errors="coerce")
    if flows.isna().any():
        bad = df.loc[flows.isna(), FORECAST_ID_COL].tolist()
        raise ValueError(f"Forecast file {path} has missing/non-numeric flows for reaches: {bad}")
    df[FORECAST_FLOW_COL] = flows.astype(float)

    return df


def validate_forecast_ids(forecast_df, catchlist_df):
    """
    Compare forecast reach IDs with the reaches of the current HAND run.

    Returns:
        dict with ``unknown`` (forecast IDs not in catchlist), ``unforecast``
        (catchlist IDs with no forecast) and ``matched`` (common IDs), each a
        sorted list
    """
    f_ids = set(forecast_df[FORECAST_ID_COL])
    c_ids = set(catchlist_df["Id"].astype(int))
    result = {
        "unknown": sorted(f_ids - c_ids),
        "unforecast": sorted(c_ids - f_ids),
        "matched": sorted(f_ids & c_ids),
    }

    if result["unknown"]:
        logger.warning(
            f"Forecast reach IDs not in catchlist.csv (stale LINKNOs from a different "
            f"outlet configuration?): {result['unknown']}"
        )
    if result["unforecast"]:
        logger.warning(
            f"Reaches with no forecast flow (no depth will be mapped there): {result['unforecast']}"
        )
    return result


def validate_rating_curve_range(forecast_df, hydroprop_df):
    """
    Check each forecast flow against the reach's synthetic rating curve.

    A flow above the largest tabulated discharge means inundepth must cap or
    extrapolate the stage, which under-predicts depth for large (e.g.
    dam-break) floods.

    Returns:
        list of dicts ``{Id, flow_m3s, max_flow_m3s, max_stage_m}`` for flows
        exceeding the rating curve
    """
    src = hydroprop_df.groupby(HYDROPROP_ID_COL).agg(
        max_flow_m3s=(HYDROPROP_FLOW_COL, "max"),
        max_stage_m=(HYDROPROP_STAGE_COL, "max"),
    )

    exceed = []
    for _, row in forecast_df.iterrows():
        reach_id = int(row[FORECAST_ID_COL])
        flow = float(row[FORECAST_FLOW_COL])
        if flow <= 0:
            logger.warning(f"Reach {reach_id}: non-positive forecast flow {flow} m3s")
            continue
        if reach_id not in src.index:
            continue
        max_q = float(src.at[reach_id, "max_flow_m3s"])
        max_h = float(src.at[reach_id, "max_stage_m"])
        if flow > max_q:
            exceed.append({"Id": reach_id, "flow_m3s": flow, "max_flow_m3s": max_q, "max_stage_m": max_h})
            logger.warning(
                f"Reach {reach_id}: forecast {flow:,.1f} m3/s exceeds rating-curve max "
                f"{max_q:,.1f} m3/s (stage {max_h:g} m). Add higher stages to stage.txt "
                f"and re-run dam_hydroprop."
            )
    return exceed


class DamInundationAnalysis:
    """Class for performing dam inundation analysis."""

    def __init__(
        self,
        taudem_path="$HOME/taudem",
        output_dir="outputs",
        hand_raster=None,
        catch_raster=None,
        forecast_file=None,
        hydroprop_file=None,
        catchlist=None,
        inundation_raster=None,
        depth_table=None,
        np=4,
    ):
        """
        Initialize the inundation analysis.

        Args:
            taudem_path: Path to TauDEM installation
            output_dir: Directory containing HAND workflow outputs; also the
                working directory and the default location of every file below
            hand_raster: HAND raster (default: <output_dir>/hand.tif)
            catch_raster: Reach catchment raster (default: <output_dir>/w2.tif)
            forecast_file: Forecast flows CSV (default: <output_dir>/forecast.csv)
            hydroprop_file: Hydraulic properties table (default: <output_dir>/hydroprop.csv)
            catchlist: Reach list CSV, used for ID checks (default: <output_dir>/catchlist.csv)
            inundation_raster: Output depth raster (default: <output_dir>/inundepth.tif)
            depth_table: Output depth summary (default: <output_dir>/depths.csv)
            np: Number of MPI processes
        """
        self.taudem_path = Path(os.path.expandvars(taudem_path)).expanduser().resolve()
        self.output_dir = Path(output_dir).resolve()

        def _resolve(path, default_name):
            return Path(path).resolve() if path else self.output_dir / default_name

        self.hand_raster = _resolve(hand_raster, "hand.tif")
        self.catch_raster = _resolve(catch_raster, "w2.tif")
        self.forecast_file = _resolve(forecast_file, "forecast.csv")
        self.hydroprop_file = _resolve(hydroprop_file, "hydroprop.csv")
        self.catchlist = _resolve(catchlist, "catchlist.csv")
        self.inundation_raster = _resolve(inundation_raster, "inundepth.tif")
        self.depth_table = _resolve(depth_table, "depths.csv")
        self.np = np

        # Set up TauDEM PATH
        setup_taudem_path(str(self.taudem_path))

        # Verify required input files exist
        self.verify_inputs()

    def verify_inputs(self):
        """Verify that all required input files exist."""
        missing_files = [
            str(f) for f in (self.hand_raster, self.catch_raster) if not f.exists()
        ]
        if missing_files:
            logger.error(f"Missing required input files: {missing_files}")
            logger.error("Please run the HAND dam workflow (dam_hand) first to generate these files.")
            sys.exit(1)

        if not self.hydroprop_file.exists():
            logger.error(f"Hydraulic properties table not found: {self.hydroprop_file}")
            logger.error("Please run dam_hydroprop first to generate it.")
            sys.exit(1)

        if not self.forecast_file.exists():
            logger.error(f"Forecast file not found: {self.forecast_file}")
            sys.exit(1)

        logger.info("All required input files found.")

    def validate_inputs(self):
        """Validate the forecast against the HAND run and the rating curve; exit on fatal problems."""
        try:
            forecast = load_forecast(self.forecast_file)
        except ValueError as e:
            logger.error(str(e))
            sys.exit(1)
        logger.info(f"Loaded forecast for {len(forecast)} reaches from {self.forecast_file}")

        if self.catchlist.exists():
            ids = validate_forecast_ids(forecast, _read_csv(self.catchlist))
            if not ids["matched"]:
                logger.error(
                    "No forecast reach IDs match catchlist.csv; the forecast belongs to a "
                    "different HAND run."
                )
                sys.exit(1)
        else:
            logger.warning(f"catchlist.csv not found ({self.catchlist}); skipping reach-ID check")

        hydroprop = _read_csv(self.hydroprop_file)
        needed = {HYDROPROP_ID_COL, HYDROPROP_STAGE_COL, HYDROPROP_FLOW_COL}
        if needed.issubset(hydroprop.columns):
            validate_rating_curve_range(forecast, hydroprop)
        else:
            logger.warning(
                f"Unexpected hydroprop.csv columns {list(hydroprop.columns)}; "
                "skipping rating-curve range check"
            )

    def run_inundation_depth(self):
        """Run inundation depth calculation."""
        logger.info("Running inundation depth calculation...")
        q = shlex.quote
        run_taudem_command(
            (
                f"mpiexec -n {int(self.np)} inundepth "
                f"-hand {q(str(self.hand_raster))} "
                f"-catch {q(str(self.catch_raster))} "
                f"-fc {q(str(self.forecast_file))} "
                f"-hp {q(str(self.hydroprop_file))} "
                f"-inun {q(str(self.inundation_raster))} "
                f"-depth {q(str(self.depth_table))}"
            ),
            "Inundation depth calculation",
            cwd=str(self.output_dir),
        )

    def run_analysis(self):
        """Execute the inundation analysis."""
        logger.info("Starting Dam Inundation Analysis...")

        try:
            self.validate_inputs()
            self.run_inundation_depth()

            logger.info("Dam Inundation Analysis completed successfully!")
            logger.info(f"- Inundation depth raster: {self.inundation_raster}")
            logger.info(f"- Inundation depth summary: {self.depth_table}")

        except Exception as e:
            logger.error(f"Inundation analysis failed: {str(e)}")
            sys.exit(1)


def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(
        description=(
            "Dam Inundation Analysis - map inundation depth for a forecast using HAND "
            "and the hydraulic properties table from dam_hydroprop"
        )
    )

    parser.add_argument(
        "--taudem-path",
        default="$HOME/taudem",
        help="Path to TauDEM installation (default: $HOME/taudem)"
    )
    parser.add_argument(
        "--dam-dir",
        default=None,
        help="Per-dam input directory (e.g. data/<dam>); only used to pick the default --output-dir"
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory containing HAND workflow outputs; default location of all files below "
             "(default: outputs/<dam-dir name> with --dam-dir, else outputs)"
    )
    parser.add_argument("--hand-raster", help="HAND raster (default: <output-dir>/hand.tif)")
    parser.add_argument("--catch-raster", help="Reach catchment raster (default: <output-dir>/w2.tif)")
    parser.add_argument(
        "--forecast-file",
        help=f"Forecast CSV with columns '{FORECAST_ID_COL}, {FORECAST_FLOW_COL}' (default: <output-dir>/forecast.csv)"
    )
    parser.add_argument("--hydroprop-file", help="Hydraulic properties table (default: <output-dir>/hydroprop.csv)")
    parser.add_argument("--catchlist", help="Reach list CSV for reach-ID checks (default: <output-dir>/catchlist.csv)")
    parser.add_argument("--inundation-raster", help="Output inundation depth raster (default: <output-dir>/inundepth.tif)")
    parser.add_argument("--depth-table", help="Output depth summary CSV (default: <output-dir>/depths.csv)")
    parser.add_argument("--np", type=int, default=4, help="Number of MPI processes (default: 4)")
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.output_dir is None:
        args.output_dir = default_output_dir(args.dam_dir) if args.dam_dir else "outputs"

    DamInundationAnalysis(
        taudem_path=args.taudem_path,
        output_dir=args.output_dir,
        hand_raster=args.hand_raster,
        catch_raster=args.catch_raster,
        forecast_file=args.forecast_file,
        hydroprop_file=args.hydroprop_file,
        catchlist=args.catchlist,
        inundation_raster=args.inundation_raster,
        depth_table=args.depth_table,
        np=args.np,
    ).run_analysis()


if __name__ == "__main__":
    main()
