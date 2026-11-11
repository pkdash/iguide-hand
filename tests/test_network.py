"""Live checks against the USGS services dam_setup depends on. Run with: pytest -m network"""

import pytest

import dam_setup

pytestmark = pytest.mark.network

# Mountain Dell Dam, the notebook's example
MOUNTAIN_DELL = (40.750594, -111.721088)


def test_wbd_lookup_mountain_dell():
    assert dam_setup.find_hu4_for_point(*MOUNTAIN_DELL) == "1602"


def test_tnm_bucket_lists_hu4_1602():
    urls = dam_setup.nhdplus_hr_urls("1602")
    assert urls["gdb"][0].endswith("/GDB/NHDPLUS_H_1602_HU4_20220412_GDB.zip")
    assert urls["raster"][0].endswith("/Raster/NHDPLUS_H_1602_HU4_20220412_RASTER.zip")
    assert urls["raster"][1] > 1e9  # the raster zip is GB-sized
