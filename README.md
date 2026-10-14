# glaciermatch

AI-generated code to do image matching on satellite glacier images.

Retrieve glacier surface motion from two satellite images of the same area by
cross-correlation feature tracking. Pick sample points on the first image, and the tool
finds where each one moved to in the second image, using normalised cross-correlation
(NCC) with a sliding window over an image pyramid (coarse-to-fine).

## Installation

```bash
conda env create -f environment.yml
conda activate glaciermatch
```

## Usage

```bash
python main.py                      # then File -> Open image 1 / Open image 2
python main.py image1.tif image2.tif
```

1. **Open both GeoTIFFs.** Input checks run immediately (georeferencing, CRS, pixel size,
   orientation, overlap, nodata, sub-pixel grid offset) and are listed with ✔ / ⚠ / ✖.
2. **Co-register.** Both images are put on image 1's pixel grid, cropped to their overlap
   (see below). Both views then show the same ground area pixel for pixel.
3. **Click points on image 1.** Left-click adds a point, right-click on a marker removes it
   (or press `Delete` to remove the selected one). The selected point shows its sample
   window on both images and its search window (dashed) on image 2.
4. **Set parameters** and press **Run matching.**
5. **Inspect** the arrows on image 2 and the results table, then **export** CSV or GeoJSON.

| Control | Action |
|---|---|
| Mouse wheel | Zoom towards the cursor |
| Drag | Pan |
| `F` | Fit both views |
| Middle double-click | Fit that view |

Zoom and pan are linked between the two views unless you untick *Link zoom and pan*.

### Parameters

| Parameter | Meaning |
|---|---|
| Sample window | Side (px) of the square patch taken from image 1. Forced odd. |
| Search window | Side (px) of the square area of image 2 searched at the coarsest level. Must be larger than the sample. |
| Pyramid levels | Number of resolution levels (each is half the previous). Increase for larger displacements. Reduced automatically if the coarsest image would be smaller than the search window. |
| Refine radius | At finer levels only ±this many px around the upsampled coarse estimate are searched. |
| Min. correlation | Matches with a peak NCC below this are flagged `low_corr`. |
| Min. peak ratio | Peak divided by the best correlation outside a small exclusion zone. Below this the match is flagged `ambiguous`. |
| Sub-pixel refinement | Parabolic fit around the correlation peak at full resolution. |
| Time interval | Days between the images; needed for velocities. |
| Arrow scale | Display magnification of the vectors only. |

The maximum displacement the search can capture at full resolution is roughly
`(search − sample) / 2 · 2^(levels − 1)` px.

## How it works

**Co-registration** (`coregistration.py`). Image 1's pixel lattice is the reference grid,
cropped to the overlap with image 2. Image 1 is not interpolated. Image 2 is copied
without interpolation when its pixel lattice is aligned with image 1's; otherwise (or if
its CRS differs) it is resampled bilinearly. Pixels without data in either image are masked
and never matched.

**Matching** (`matching.py`, `multiscale.py`). For each point the sample patch is slid over
the search window and scored with zero-mean NCC (`cv2.matchTemplate`). The peak gives the
integer offset, refined to sub-pixel precision. With a pyramid, matching starts on the
coarsest level with the full search window; each estimate is doubled and used to centre a
small search at the next finer level. Window sizes are constant in pixels, so they cover
more ground at coarse levels.

**Quality flags** (the `flags` column; an empty value means `ok`):

| Flag | Meaning |
|---|---|
| `low_corr` | Peak correlation below the threshold |
| `ambiguous` | Second peak almost as high as the first |
| `border` | Peak on the edge of the search window: the true shift may lie outside it |
| `level_inconsistent` | A fine level moved far from the coarser estimate |
| `low_texture` | Sample patch is (almost) uniform, e.g. fresh snow |
| `masked` | Sample or search window touches no-data pixels |
| `outside` | Window falls outside the image |

## Output

CSV and GeoJSON, one row/feature per point, located at the sample position in the CRS of
image 1:

| Column | Meaning |
|---|---|
| `id`, `col`, `row` | Point number and pixel position on the common grid |
| `x`, `y` | Map coordinates |
| `dx_px`, `dy_px` | Displacement in pixels, image convention (+y is down) |
| `dx_m`, `dy_m` | Displacement in metres, east and north (projected CRS only) |
| `disp_px`, `disp_m` | Displacement magnitude |
| `vel_m_day`, `vel_m_yr` | Velocity (needs the time interval) |
| `corr`, `peak_ratio` | Match quality |
| `ok`, `flags` | Whether all checks passed, and which failed |

Flagged points are still reported; filter on `ok`.

## Assumptions and limitations

- Both images must be georeferenced GeoTIFFs from the same sensor (same resolution).
  A different CRS or pixel size is handled with a warning, but resampling smooths image 2.
- Residual geolocation error between the scenes appears as spurious motion. Stable-ground
  calibration is not implemented yet: check bedrock points by eye.
- Matching assumes the glacier surface keeps its appearance between the dates. Strong
  changes (snowfall, melt, shadows) will give low correlation.
- Only one band is used (band 1).
- The whole overlap is loaded into memory and shown as a single image, so very large
  scenes will be slow.
- The time interval is entered by hand.

## Development

```bash
python -m pytest        # synthetic GeoTIFFs; the GUI test runs offscreen
```

```
glaciermatch/
├── io.py             # GeoTIFF metadata
├── checks.py         # input validation
├── coregistration.py # common grid, resampling, valid mask
├── pyramid.py        # Gaussian pyramid, coordinate scaling
├── matching.py       # NCC, peak finding, sub-pixel, flags
├── multiscale.py     # coarse-to-fine driver
├── results.py        # results table, CSV/GeoJSON export
└── gui/              # PySide6 interface
```
