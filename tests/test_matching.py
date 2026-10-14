import numpy as np
import pytest
from scipy.ndimage import shift as ndshift

from glaciermatch.matching import match_point, ncc_surface, ncc_surface_fft
from glaciermatch.multiscale import match_points
from glaciermatch.pyramid import build_pyramid, from_level, to_level


def shifted(img, dx, dy):
    # a feature at x in img appears at x + dx in the output
    return ndshift(img, (dy, dx), order=3, mode="reflect")


def test_pyramid_shapes_and_coords():
    img = np.random.rand(100, 130).astype(np.float32)
    p = build_pyramid(img, 4)
    assert [l.shape for l in p] == [(100, 130), (50, 65), (25, 33), (13, 17)]
    assert from_level(to_level(80, 3), 3) == 80
    assert len(build_pyramid(img, 10, min_size=40)) == 2


def test_ncc_matches_fft_reference(texture):
    t = texture[100:131, 100:131]
    s = texture[80:150, 90:160]
    np.testing.assert_allclose(ncc_surface(t, s), ncc_surface_fft(t, s), atol=1e-3)


@pytest.mark.parametrize("dx,dy", [(3.0, -2.0), (2.3, 1.6), (-4.75, 0.25)])
def test_subpixel_recovery(texture, dx, dy):
    img2 = shifted(texture, dx, dy)
    r = match_point(texture, img2, 256, 256, 31, 61)
    assert r.ok
    assert abs(r.dx - dx) < 0.15 and abs(r.dy - dy) < 0.15
    assert r.corr > 0.95


def test_integer_shift_exact_without_subpixel(texture):
    img2 = shifted(texture, 5, -3)
    r = match_point(texture, img2, 200, 300, 31, 61, subpixel=False)
    assert (r.dx, r.dy) == (5, -3)


def test_border_flag_when_shift_exceeds_search(texture):
    img2 = shifted(texture, 25, 0)
    r = match_point(texture, img2, 256, 256, 31, 51)  # +-10 px range
    assert "border" in r.flags


def test_low_texture_flag():
    flat = np.full((200, 200), 7.0, dtype=np.float32)
    r = match_point(flat, flat, 100, 100, 31, 61)
    assert "low_texture" in r.flags and not r.ok


def test_outside_flag(texture):
    r = match_point(texture, texture, 5, 5, 31, 61)
    assert "outside" in r.flags


def test_masked_flag(texture):
    valid = np.ones(texture.shape, bool)
    valid[250:260, 250:260] = False
    r = match_point(texture, texture, 256, 256, 31, 61, valid=valid)
    assert "masked" in r.flags


def test_multiscale_recovers_large_shift(texture):
    dx, dy = 40.0, -24.0
    img2 = shifted(texture, dx, dy)
    pts = [(200, 300), (300, 250)]
    # single scale with a small window cannot reach it
    single = match_points(texture, img2, pts, 21, 41, n_levels=1)
    assert all(not r.ok for r in single)
    multi = match_points(texture, img2, pts, 21, 41, n_levels=4)
    for r in multi:
        assert r.ok, r.flags
        assert abs(r.dx - dx) < 0.2 and abs(r.dy - dy) < 0.2


def test_multiscale_fractional_shift(texture):
    img2 = shifted(texture, 7.4, 3.7)
    (r,) = match_points(texture, img2, [(256, 256)], 31, 61, n_levels=3)
    assert r.ok and abs(r.dx - 7.4) < 0.15 and abs(r.dy - 3.7) < 0.15


def test_multiscale_flags_low_corr(texture):
    other = np.roll(texture, 17, axis=0)[::-1].copy()  # unrelated content
    (r,) = match_points(texture, other, [(256, 256)], 31, 61, n_levels=2, min_corr=0.8)
    assert not r.ok
