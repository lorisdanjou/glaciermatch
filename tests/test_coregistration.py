import numpy as np
import pytest
import rasterio
from affine import Affine
from rasterio.warp import Resampling, reproject

from glaciermatch.checks import Status, check_inputs
from glaciermatch.coregistration import CoregistrationError, coregister
from glaciermatch.io import read_info

from conftest import write_tif

ORIGIN = (500000.0, 5000000.0)


def crop(world, c0, r0, w, h):
    return world[r0:r0 + h, c0:c0 + w]


def origin_at(c, r, res=10.0, frac=(0.0, 0.0)):
    return (ORIGIN[0] + (c + frac[0]) * res, ORIGIN[1] - (r + frac[1]) * res)


def statuses(checks):
    return {c.name: c.status for c in checks}


def test_integer_aligned_crop(tmp_path, world):
    a = write_tif(tmp_path / "a.tif", crop(world, 0, 0, 300, 300), origin_at(0, 0))
    b = write_tif(tmp_path / "b.tif", crop(world, 100, 50, 300, 300), origin_at(100, 50))
    r = coregister(a, b)
    assert r.shape == (250, 200)
    assert r.resampling == "nearest"
    assert statuses(r.checks)["lattice_offset"] is Status.OK
    expected = world[50:300, 100:300]
    np.testing.assert_array_equal(r.img1, expected)
    np.testing.assert_array_equal(r.img2, expected)  # exact copy, no interpolation
    assert r.valid.all()
    # common grid origin is where image 2 starts
    assert r.pix_to_map(0, 0) == (origin_at(100, 50)[0] + 5, origin_at(100, 50)[1] - 5)


def test_image2_extent_is_larger(tmp_path, world):
    a = write_tif(tmp_path / "a.tif", crop(world, 100, 100, 100, 100), origin_at(100, 100))
    b = write_tif(tmp_path / "b.tif", world, origin_at(0, 0))
    r = coregister(a, b)
    assert r.shape == (100, 100)
    np.testing.assert_array_equal(r.img2, r.img1)


def test_fractional_offset_interpolates(tmp_path, world):
    a = write_tif(tmp_path / "a.tif", crop(world, 0, 0, 300, 300), origin_at(0, 0))
    b = write_tif(tmp_path / "b.tif", crop(world, 50, 50, 300, 300), origin_at(50, 50, frac=(0.5, 0.25)))
    r = coregister(a, b)
    assert r.resampling == "bilinear"
    assert statuses(r.checks)["lattice_offset"] is Status.WARN
    assert abs(r.frac_offset[0]) == pytest.approx(0.5) and r.frac_offset[1] == pytest.approx(0.25)
    assert r.valid.mean() > 0.95
    # image 1 untouched: the grid starts at the first whole pixel inside image 2 (51, 51)
    ys, xs = r.shape
    assert np.array_equal(r.img1, world[51:51 + ys, 51:51 + xs])


def test_no_overlap_raises(tmp_path, world):
    a = write_tif(tmp_path / "a.tif", crop(world, 0, 0, 100, 100), origin_at(0, 0))
    b = write_tif(tmp_path / "b.tif", crop(world, 200, 200, 100, 100), origin_at(200, 200))
    with pytest.raises(CoregistrationError, match="do not overlap"):
        coregister(a, b)


def test_missing_georeferencing_raises(tmp_path, world):
    a = write_tif(tmp_path / "a.tif", crop(world, 0, 0, 100, 100), origin_at(0, 0))
    b = tmp_path / "b.tif"
    with rasterio.open(b, "w", driver="GTiff", height=100, width=100, count=1, dtype="float32") as ds:
        ds.write(crop(world, 0, 0, 100, 100), 1)
    with pytest.raises(CoregistrationError, match="no CRS"):
        coregister(a, str(b))


def test_nodata_mask(tmp_path, world):
    a = write_tif(tmp_path / "a.tif", crop(world, 0, 0, 200, 200), origin_at(0, 0), nodata=-9999.0)
    bd = crop(world, 0, 0, 200, 200).copy()
    bd[40:60, 70:90] = -9999.0
    b = write_tif(tmp_path / "b.tif", bd, origin_at(0, 0), nodata=-9999.0)
    r = coregister(a, b)
    assert not r.valid[40:60, 70:90].any()
    assert r.valid.sum() == 200 * 200 - 400
    a1, a2 = r.matching_arrays()
    assert np.isfinite(a1).all() and np.isfinite(a2).all()


def test_crs_mismatch_reprojects(tmp_path, world):
    a = write_tif(tmp_path / "a.tif", crop(world, 0, 0, 300, 300), origin_at(0, 0), crs="EPSG:32633")
    # build image 2 in UTM 32N by reprojecting a larger piece of the same ground
    src = write_tif(tmp_path / "src.tif", world, origin_at(0, 0), crs="EPSG:32633")
    dst_path = tmp_path / "b.tif"
    from rasterio.warp import calculate_default_transform
    with rasterio.open(src) as ds:
        t, w, h = calculate_default_transform(ds.crs, "EPSG:32632", ds.width, ds.height, *ds.bounds,
                                              resolution=10.0)
        out = np.zeros((h, w), np.float32)
        reproject(rasterio.band(ds, 1), out, dst_transform=t, dst_crs="EPSG:32632",
                  resampling=Resampling.bilinear)
    write_tif(dst_path, out, crs="EPSG:32632", transform=t)
    r = coregister(a, str(dst_path))
    st = statuses(r.checks)
    assert st["crs"] is Status.WARN
    assert r.valid.mean() > 0.9
    m = r.valid
    c = np.corrcoef(r.img1[m], r.img2[m])[0, 1]
    assert c > 0.98


def test_pixel_size_mismatch_warns(tmp_path, world):
    a = write_tif(tmp_path / "a.tif", crop(world, 0, 0, 200, 200), origin_at(0, 0))
    b = write_tif(tmp_path / "b.tif", crop(world, 0, 0, 100, 100), origin_at(0, 0), res=20.0)
    r = coregister(a, b)
    assert statuses(r.checks)["pixel_size"] is Status.WARN
    assert r.shape == (200, 200)


def test_checks_report_without_running(tmp_path, world):
    a = write_tif(tmp_path / "a.tif", crop(world, 0, 0, 100, 100), origin_at(0, 0))
    b = write_tif(tmp_path / "b.tif", crop(world, 0, 0, 100, 100), origin_at(0, 0))
    checks = check_inputs(read_info(a), read_info(b))
    assert all(c.status in (Status.OK, Status.WARN) for c in checks)
    assert statuses(checks)["nodata"] is Status.WARN


def test_rotated_image1_is_warped_north_up(tmp_path, world):
    rot = Affine(10, 1.0, ORIGIN[0], 1.0, -10, ORIGIN[1])  # slight shear
    a = write_tif(tmp_path / "a.tif", crop(world, 0, 0, 200, 200), transform=rot)
    b = write_tif(tmp_path / "b.tif", crop(world, 0, 0, 200, 200), origin_at(0, 0))
    r = coregister(a, b)
    assert statuses(r.checks)["north_up"] is Status.WARN
    assert r.valid.any() and r.transform.b == 0 and r.transform.d == 0
