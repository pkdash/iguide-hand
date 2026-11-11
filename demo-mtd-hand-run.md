# HAND pipeline demo (Mountain Dell Dam)

Run all commands from the repo root, in one shell session, because later steps use the variables set in step 0. The same steps work for any dam: change `DAM` and `OUT` in step 0, and the coordinates in step 1 if you use them.

## 0. Set up the dam folder

Each dam has a self-contained folder: the dam's inputs, plus the NHDPlus HR dataset for the HU4 that contains the dam. You can build it yourself, or let `dam_catchments` build it from the dam's coordinates in step 1 (Option A there): then skip to "Then set" below.

```text
data/mountain_dell/
├── mountain_dell_dam_location.shp        # dam point (dam_location.shp or a single *dam_location.shp)
├── stage.txt
├── forecast.csv                          # optional
├── Outlets2_utm.shp                      # optional, only for Option B in step 2
├── NHDPLUS_H_1602_HU4_20220412_GDB/      # HU4 geodatabase (found automatically)
└── NHDPLUS_H_1602_HU4_20220412_RASTER/   # HU4 rasters with hydrodem.tif (found automatically)
```

If you still have the original flat `data/` layout, build the folder from it. The NHDPlus HR data is several GB, so symlink it rather than copying; or `mv` it into the folder if no other dam uses it:

```bash
mkdir -p data/mountain_dell
cp data/mountain_dell_dam_location.* data/stage.txt data/forecast.csv data/Outlets2_utm.* data/mountain_dell/
ln -s ../NHDPLUS_H_1602_HU4_20220412_GDB data/mountain_dell/
ln -s ../NHDPLUS_H_1602_HU4_20220412_RASTER data/mountain_dell/
```

Then set:

```bash
DAM=data/mountain_dell
OUT=outputs/mountain_dell     # default output folder: outputs/<name of $DAM>
TAUDEM=~/taudem
```

Every stage writes to `$OUT` by default, so the commands below don't pass `--output-dir`. If you want a different folder (for example `outputs-demo`), set `OUT` to it and add `--output-dir $OUT` to every command.

## 1. Downstream catchments and clipped rasters

Finds the HUC12s downstream of the dam and clips the HU4 rasters into `$OUT/clipped`. It also writes `$OUT/run_info.json`, which records the GDB and raster paths it used.

**Option A: from the dam's coordinates.** `dam_catchments` creates `$DAM` if needed and adds whatever is missing:
- `mountain_dell_dam_location.shp`, the dam point;
- `stage.txt`, with default stage heights from 0.1 to 40 m;
- the NHDPlus HR GDB and raster folder for the dam's HU4.

It looks up the HU4 (1602 here) and downloads the data into `data/`, then links it into `$DAM`. The raster zip is 1–6 GB, so allow about 3× that in free disk space. Existing files are kept: if the data is already in `$DAM` or `data/`, nothing is downloaded, and a rerun needs no network access.

```bash
uv run dam_catchments --dam-dir $DAM --dam-lat 40.750594 --dam-lon -111.721088 --nhd-cache-dir data
```

Put the point on the channel just below the dam embankment: NID coordinates often fall in the reservoir. If `$DAM` already has a dam shapefile more than 10 m from these coordinates, the command stops with an error; delete that shapefile to use the new point.

**Option B: from the folder built in step 0.**

```bash
uv run dam_catchments --dam-dir $DAM
```

The log lists the `HUC12 GDB` and `Reference raster` it found. The console summary ends with `Stopped: ...`, which says why the downstream walk stopped:
- the 100 km distance limit;
- a terminal basin;
- the edge of the HU4, which also logs a warning.

Stop and check your data if:
- **the GDB or raster folder isn't found, or several are:** the dam folder is missing its NHDPlus data (use Option A to download it) or has more than one set;
- **the dam is more than 1 km from every HUC12 in the GDB:** the NHDPlus data is for the wrong HU4.

Takes about 10 seconds for Mountain Dell.

## 2. HAND (TauDEM)

Builds:
- the TauDEM grids;
- the stream network, split into reaches;
- `hand.tif` and `catchlist.csv`.

It reads the rasters in `$OUT/clipped`. It copies `stage.txt` from `$DAM`, and `forecast.csv` too if present.

**Option A: 2 km segment outlets.** The main stem ends just upstream of the first large lake or reservoir in the dam's NHDPlus GDB. For Mountain Dell that is Great Salt Lake, about 60 km below the dam; look for `Stopping where the main stem enters ...` in the log.

```bash
uv run dam_hand --dam-dir $DAM --taudem-path $TAUDEM --segment-length 2000
```

**Option B: user outlets** (`$DAM/Outlets2_utm.shp`)

```bash
uv run dam_hand --dam-dir $DAM --taudem-path $TAUDEM
```

Takes about 2–3 minutes for Mountain Dell. Keep `--np` at its default (4) between runs. The reach IDs (`LINKNO`) depend on the number of MPI processes, so changing it changes the IDs.

## 3. Synthetic rating curves

Writes `$OUT/hydroprop.csv`.

```bash
uv run dam_hydroprop --dam-dir $DAM --taudem-path $TAUDEM
```

## 4. Inundation

Writes `$OUT/inundepth.tif` and `$OUT/depths.csv`. Give the flows with exactly one of `--uniform-flow` or `--forecast-file`; the command stops with a usage error if neither is given.

Reach IDs change with the outlets, so a `forecast.csv` from `$DAM` won't match a 2 km run. Without a forecast that matches this run, map the same flow on every reach:

```bash
uv run dam_inundation --dam-dir $DAM --taudem-path $TAUDEM --uniform-flow 4000
```

This first writes `$OUT/forecast_uniform4000.csv` from this run's `$OUT/catchlist.csv`.

If you have your own forecast whose `Id` values match this run's `$OUT/catchlist.csv`, pass it instead:

```bash
uv run dam_inundation --dam-dir $DAM --taudem-path $TAUDEM --forecast-file my_forecast.csv
```

A bare file name is looked up in `$DAM`; any other path is relative to the current directory.

To show another flood scenario, change `4000` and rerun only this step. `hydroprop.csv` is reused. Add `--inundation-raster $OUT/inundepth_8000.tif --depth-table $OUT/depths_8000.csv` to keep each scenario's outputs instead of overwriting them.

## What to show

| File | What it is |
|---|---|
| `$OUT/downstream_huc12s_from_dam.shp` | HUC12s downstream of the dam (study area) |
| `$OUT/processing_report.txt` | Catchment walk summary, including why it stopped |
| `$OUT/net2.shp` | Stream network, split into reaches |
| `$OUT/hand.tif` | Height Above Nearest Drainage |
| `$OUT/catchlist.csv` | Reaches: ID, slope, length, Manning's n |
| `$OUT/hydroprop.csv` | Rating curve for each reach (stage, area, volume, discharge) |
| `$OUT/forecast_uniform4000.csv` | Flow scenario: 4,000 m³/s on every reach |
| `$OUT/inundepth.tif` | Flood depth map |
| `$OUT/depths.csv` | Water level for each reach. `InunRatio = 1` means the catchment, not the channel, caps the water level, so that depth isn't reliable. |

Add `-v` to any command for more logging, or `--help` for all options.
