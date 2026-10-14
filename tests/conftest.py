import numpy as np
import pytest
import rasterio
from affine import Affine
from scipy.ndimage import gaussian_filter


def make_texture(shape=(512, 512), seed=0):
    """Multi-scale random texture, so coarse pyramid levels keep structure."""
    rng = np.random.default_rng(seed)
    img = np.zeros(shape)
    for sigma, w in ((1.5, 1.0), (4, 1.5), (10, 2.0), (25, 2.5)):
        n = gaussian_filter(rng.standard_normal(shape), sigma)
        img += w * n / n.std()
    return img.astype(np.float32)


@pytest.fixture(scope="session")
def texture():
    return make_texture()


def write_tif(path, arr, origin=(500000.0, 5000000.0), res=10.0, crs="EPSG:32633",
              nodata=None, transform=None):
    t = transform or Affine(res, 0, origin[0], 0, -res, origin[1])
    with rasterio.open(path, "w", driver="GTiff", height=arr.shape[0], width=arr.shape[1],
                       count=1, dtype=arr.dtype, crs=crs, transform=t, nodata=nodata) as ds:
        ds.write(arr, 1)
    return str(path)


@pytest.fixture
def world():
    return make_texture((400, 500), seed=3)
