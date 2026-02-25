# ORS Assignment 2 — Cloud Removal Using PMAA

Cloud removal from multi-temporal Sentinel-2 imagery over **Sangareddy district, Telangana** using the [PMAA](https://github.com/XavierJiezou/PMAA) model (ECAI 2023).

**Team:** Goni Anagha (2023101124) · Gandlur Valli (2023102068) · Nunna Sri Abhinaya (2023102071) · Vishakha (2023101040)

## Repository Contents

| File | Description |
|------|-------------|
| `Geospatial_Analysis_Team_A2.pdf` | Final report (compiled) |
| `Geospatial_Analysis_Team_A2.tex` | LaTeX source for the report |
| `Geospatial_Analysis_Team_A2.txt` | Moodle submission text |
| `code/gee.js` | Google Earth Engine pipeline for data extraction |
| `code/pmaa_ada_inference.py` | Custom district-scale inference script (~745 lines) |
| `code/run_pmaa.sh` | SLURM submission script for Ada HPC |
| `metrics.txt` | Evaluation results (PSNR, SSIM) |
| `figures/figures.zip` | Report figures including complementarity map |
| `PMAA_paper.pdf` | Original PMAA paper (Zou et al., ECAI 2023) |
| `gee_console` | GEE console output |

### Output Visualisations (`output_images/`)

| File | Description |
|------|-------------|
| `input_rgb.png` | True-colour composites of the three cloudy inputs and reference |
| `results_comparison.png` | Full-district results with difference maps |
| `noblend_tif.png` | No-blend output (direct 256×256 tiling) |
| `blend_tif.png` | Overlap-blend output (64 px overlap, cosine weighting) |
| `blend_zoom.png` | Zoomed 512×512 centre crop comparing no-blend, blend, and reference |
| `patch_showcase.png` | Five representative patches at different quality levels |

## Data

Exported GeoTIFFs and pretrained weights are on Google Drive (too large for Git):

**[Google Drive link](https://drive.google.com/drive/folders/1Z0_m13DWVgvzmZwhDxfemVZ2GhAha7Ak?usp=drive_link)**

Contents: 3 cloudy mosaics (Aug–Oct 2024), median reference (Nov–Dec 2024), SCL mask, and `pmaa_new.pth` pretrained weights.

## Pipeline Overview

```
GEE (code/gee.js)               →  Export cloudy mosaics + reference to Drive
                                     ↓
Ada HPC (code/run_pmaa.sh)       →  Copy data to /scratch, run inference
                                     ↓
Inference script                 →  Tile merge → Patch extraction → PMAA forward
(code/pmaa_ada_inference.py)         → Overlap-blend stitching → Evaluation → Visualisation
```

### GEE Pipeline (`code/gee.js`)

- Loads Sangareddy boundary from uploaded shapefile asset
- Filters Sentinel-2 SR Harmonized: 30–85% cloud cover, Aug–Oct 2024
- Computes spatial complementarity (union clear coverage: ~73%)
- Builds Nov–Dec 2024 median reference (<10% cloud)
- Exports 4-band GeoTIFFs (B2, B3, B4, B8) at 10 m resolution

### Inference (`code/pmaa_ada_inference.py`)

- Loads PMAA with `PMAA(32, 4)` config and `pmaa_new.pth` weights
- Replaces BatchNorm → InstanceNorm for single-sample inference
- Streaming inference: images kept as uint16, patches normalised on-the-fly
- Dual mode: no-blend (overlap=0) and overlap-blend (overlap=64 px, cosine weighting)
- Evaluation: per-patch PSNR and SSIM against reference (818 valid patches)

## Results

| Metric | No-Blend | Overlap-Blend |
|--------|----------|---------------|
| PSNR (dB) | 24.35 ± 2.67 | 24.15 ± 2.10 |
| SSIM | 0.686 ± 0.208 | 0.684 ± 0.210 |
| Inference time | 100.4 s | 157.1 s |

Computed on Ada HPC (RTX 2080 Ti, 30 GB RAM) over 818 evaluated patches.

## Reproducing

1. Accept the GEE exports from the Drive link above and place them in a `data/` directory.
2. Clone the [PMAA repo](https://github.com/XavierJiezou/PMAA) and download `pmaa_new.pth`.
3. Edit paths in `code/run_pmaa.sh` and `code/pmaa_ada_inference.py` to match your setup.
4. Submit via SLURM: `sbatch code/run_pmaa.sh`

## References

- Zou et al., "PMAA: A Progressive Multi-scale Attention Autoencoder Model for High-Performance Cloud Removal from Multi-temporal Satellite Imagery," ECAI 2023. [GitHub](https://github.com/XavierJiezou/PMAA)
- Sarukkai et al., "Cloud Removal from Satellite Images Using Spatiotemporal Generator Networks," WACV 2020.