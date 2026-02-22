#!/usr/bin/env python3
"""
PMAA Cloud Removal — Generalised for any district.
Memory-efficient version for Ada HPC.

Runs TWO inference passes:
  (A) No-blend  — overlap=0, direct stitching (fast, shows block artifacts)
  (B) Blend     — overlap=64, cosine-weighted blending (smooth)
Produces comparison metrics, full-image visualisations, and per-patch showcase.
"""

import os, sys, math, glob, re, argparse, subprocess, warnings, time
from typing import List, Tuple, Dict, Optional
import numpy as np
warnings.filterwarnings('ignore')


# ═════════════════════════════════════════════════════════════════════
# CONFIG
# ═════════════════════════════════════════════════════════════════════

CONFIG = {
    'patch_size': 256,
    'overlap': 64,
    'hidden_channels': 32,
    'out_channels': 4,
    'norm_max': 10000.0,
    'model_filename': 'pmaa_new.pth',
    'batch_size': 8,
}

# ═════════════════════════════════════════════════════════════════════
# DISTRICT CONFIG — change these for each district
# ═════════════════════════════════════════════════════════════════════

DISTRICT = 'Sangareddy'
DISTRICT_PREFIX = 'sangareddy'   # filename prefix (lowercase, no spaces)

CLOUDY_KEYS = ['oct2024', 'aug2024', 'sep2024']
REF_KEY = 'novdec2024med'

LABELS = {
    'oct2024':        'Oct 2024 (Cloudy)',
    'aug2024':        'Aug 2024 (Cloudy)',
    'sep2024':        'Sep 2024 (Cloudy)',
    'novdec2024med':  'Nov-Dec 2024 (Median Ref)',
}

# # ── Belgaum ──
# DISTRICT = 'Belgaum'
# DISTRICT_PREFIX = 'belgaum'
# CLOUDY_KEYS = ['jul2025', 'aug2025', 'sep2025']
# REF_KEY = 'may2025med'
# LABELS = {
#     'jul2025':    'July 2025 (Mosaic)',
#     'aug2025':    'August 2025 (Mosaic)',
#     'sep2025':    'September 2025 (Mosaic)',
#     'may2025med': 'May 2025 (Median Ref)',
# }


# ═════════════════════════════════════════════════════════════════════
# GeoTIFF I/O
# ═════════════════════════════════════════════════════════════════════

def load_geotiff(filepath):
    import rasterio
    with rasterio.open(filepath) as src:
        data = src.read()
        meta = {
            'crs': src.crs, 'transform': src.transform,
            'width': src.width, 'height': src.height,
            'count': src.count, 'dtype': src.dtypes[0],
            'nodata': src.nodata,
        }
    return data, meta


def save_geotiff(filepath, data, meta):
    import rasterio
    C, H, W = data.shape
    out_meta = {
        'driver': 'GTiff', 'height': H, 'width': W, 'count': C,
        'dtype': data.dtype, 'crs': meta['crs'],
        'transform': meta['transform'], 'compress': 'lzw',
    }
    if meta.get('nodata') is not None:
        out_meta['nodata'] = meta['nodata']
    os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
    with rasterio.open(filepath, 'w', **out_meta) as dst:
        dst.write(data)
    print(f'  Saved: {os.path.basename(filepath)}  {data.shape}  {data.dtype}', flush=True)


def read_patch_from_file(filepath, y, x, ps):
    """Read a single patch via rasterio windowed reading (memory-efficient)."""
    import rasterio
    from rasterio.windows import Window
    with rasterio.open(filepath) as src:
        window = Window(col_off=x, row_off=y, width=ps, height=ps)
        return src.read(window=window)


def merge_gee_tiles(data_dir):
    """Merge GEE tiled exports using rasterio."""
    import rasterio
    from rasterio.merge import merge as rio_merge
    tile_pattern = re.compile(r'^(.+)-(\d{10})-(\d{10})\.tif$')
    all_files = sorted(glob.glob(os.path.join(data_dir, '*.tif')))
    tile_groups = {}
    for f in all_files:
        m = tile_pattern.match(os.path.basename(f))
        if m:
            base = m.group(1)
            if base not in tile_groups:
                tile_groups[base] = []
            tile_groups[base].append(f)
    if not tile_groups:
        print('  No tiled exports found.', flush=True)
        return
    for base, tiles in tile_groups.items():
        merged = os.path.join(data_dir, base + '.tif')
        if os.path.exists(merged):
            print(f'  Already merged: {os.path.basename(merged)}', flush=True)
            continue
        print(f'  Merging {len(tiles)} tiles -> {os.path.basename(merged)}', flush=True)
        src_files = [rasterio.open(t) for t in sorted(tiles)]
        mosaic, out_transform = rio_merge(src_files)
        out_meta = src_files[0].meta.copy()
        out_meta.update({
            'height': mosaic.shape[1], 'width': mosaic.shape[2],
            'transform': out_transform, 'compress': 'lzw',
        })
        for s in src_files:
            s.close()
        with rasterio.open(merged, 'w', **out_meta) as dst:
            dst.write(mosaic)
        print(f'  Done: {os.path.basename(merged)} ({mosaic.shape})', flush=True)


def discover_exports(data_dir):
    tile_re = re.compile(r'-\d{10}-\d{10}\.tif$')
    files = sorted(glob.glob(os.path.join(data_dir, DISTRICT_PREFIX + '_*.tif')))
    files = [f for f in files if not tile_re.search(os.path.basename(f))]
    known_keys = CLOUDY_KEYS + [REF_KEY]
    groups = {}
    for f in files:
        basename = os.path.basename(f).replace('.tif', '')
        for key in known_keys:
            prefix = DISTRICT_PREFIX + '_' + key + '_'
            if basename.startswith(prefix):
                suffix = basename[len(prefix):]
                if key not in groups:
                    groups[key] = {}
                if 'B2B3B4B8' in suffix:
                    groups[key]['bands'] = f
                elif 'SCL' in suffix:
                    groups[key]['scl'] = f
                break
    return groups


# ═════════════════════════════════════════════════════════════════════
# NORMALISATION
# ═════════════════════════════════════════════════════════════════════

def normalize_s2(img, norm_max=10000.0):
    return (img.astype(np.float32) / norm_max - 0.5) / 0.5

def denormalize_s2(img, norm_max=10000.0):
    return np.clip(img * 0.5 + 0.5, 0, 1) * norm_max


# ═════════════════════════════════════════════════════════════════════
# MODEL
# ═════════════════════════════════════════════════════════════════════

def replace_batchnorm(model):
    import torch.nn as nn
    for name, child in model.named_children():
        if isinstance(child, nn.BatchNorm2d):
            setattr(model, name, nn.InstanceNorm2d(child.num_features))
        else:
            replace_batchnorm(child)


def load_pmaa_model(model_path, hidden_channels=32, out_channels=4, device='cpu'):
    import torch
    from model.pmaa import PMAA
    model = PMAA(hidden_channels, out_channels)
    replace_batchnorm(model)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device).eval()
    n = sum(p.numel() for p in model.parameters())
    print(f'  PMAA loaded: {n:,} params on {device}', flush=True)
    return model


# ═════════════════════════════════════════════════════════════════════
# MEMORY-EFFICIENT STREAMING INFERENCE
# ═════════════════════════════════════════════════════════════════════

def compute_patch_grid(H, W, ps, ov):
    stride = ps - ov
    nr = max(1, math.ceil((H - ov) / stride)) if ov > 0 else max(1, math.ceil(H / stride))
    nc = max(1, math.ceil((W - ov) / stride)) if ov > 0 else max(1, math.ceil(W / stride))
    Hp = nr * stride + ov if ov > 0 else nr * ps
    Wp = nc * stride + ov if ov > 0 else nc * ps
    coords = []
    for r in range(nr):
        for c in range(nc):
            coords.append((r * stride, c * stride))
    return coords, (Hp, Wp), nr, nc


def pad_image(img, Hp, Wp):
    C, H, W = img.shape
    if H == Hp and W == Wp:
        return img
    padded = np.zeros((C, Hp, Wp), dtype=img.dtype)
    padded[:, :H, :W] = img
    return padded


def extract_single_patch(img_padded, y, x, ps):
    return img_padded[:, y:y+ps, x:x+ps]


def make_cosine_weight(ps, ov):
    if ov <= 0:
        return np.ones((ps, ps), dtype=np.float32)
    w = np.ones(ps, dtype=np.float32)
    ramp = 0.5 * (1.0 - np.cos(np.pi * np.arange(ov) / ov))
    w[:ov] = ramp; w[-ov:] = ramp[::-1]
    return np.outer(w, w)


def run_inference_streaming(model, cloudy_imgs, config, device='cpu',
                            overlap_override=None, label=''):
    """Stream-based inference with configurable overlap."""
    import torch
    ps = config['patch_size']
    ov = overlap_override if overlap_override is not None else config['overlap']
    bs = config['batch_size']
    nm = config['norm_max']
    nch = config['out_channels']
    C, H, W = cloudy_imgs[0].shape
    tag = f'[{label}]' if label else ''

    coords, (Hp, Wp), nr, nc = compute_patch_grid(H, W, ps, ov)
    npatches = len(coords)
    print(f'  {tag} Grid: {nr}x{nc} = {npatches} patches, ps={ps}, ov={ov}', flush=True)

    padded = [pad_image(img, Hp, Wp) for img in cloudy_imgs]
    acc = np.zeros((nch, Hp, Wp), dtype=np.float32)
    wacc = np.zeros((1, Hp, Wp), dtype=np.float32)
    w2d = make_cosine_weight(ps, ov)

    t0 = time.time()
    with torch.no_grad():
        for start in range(0, npatches, bs):
            end = min(start + bs, npatches)
            batch_list = []
            for i in range(start, end):
                y, x = coords[i]
                temporal = []
                for t in range(3):
                    patch = extract_single_patch(padded[t], y, x, ps)
                    patch_norm = normalize_s2(patch, nm)
                    temporal.append(patch_norm)
                stacked = np.stack(temporal, axis=0)
                batch_list.append(stacked)

            batch_np = np.stack(batch_list, 0)
            batch_t = torch.from_numpy(batch_np).float().to(device)
            out_t, _, _ = model(batch_t)
            out_np = out_t.cpu().numpy()

            for b_idx in range(out_np.shape[0]):
                patch_out = denormalize_s2(out_np[b_idx], nm)
                y, x = coords[start + b_idx]
                acc[:, y:y+ps, x:x+ps] += patch_out * w2d[np.newaxis]
                wacc[:, y:y+ps, x:x+ps] += w2d[np.newaxis]

            elapsed = time.time() - t0
            pct = 100 * end / npatches
            eta = elapsed / end * (npatches - end) if end > 0 else 0
            print(f'\r  {tag} [{int(pct):3d}%] {end}/{npatches}  '
                  f'elapsed={elapsed:.0f}s  ETA={eta:.0f}s', end='', flush=True)

    print(flush=True)
    result = acc / np.maximum(wacc, 1e-8)
    return result[:, :H, :W]


# ═════════════════════════════════════════════════════════════════════
# EVALUATION
# ═════════════════════════════════════════════════════════════════════

def compute_psnr(a, b, dr=10000.0):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64))**2)
    return 10*np.log10(dr**2/mse) if mse > 1e-10 else float('inf')


def compute_ssim(a, b, dr=10000.0):
    from skimage.metrics import structural_similarity as ssim_fn
    return ssim_fn(a.transpose(1,2,0).astype(np.float64),
                   b.transpose(1,2,0).astype(np.float64),
                   channel_axis=-1, data_range=dr,
                   gaussian_weights=True, use_sample_covariance=False, sigma=1.5)


def create_scl_cloud_mask(scl_data, target_shape):
    from scipy.ndimage import zoom as scipy_zoom
    scl = scl_data.squeeze()
    Ht, Wt = target_shape
    if scl.shape != (Ht, Wt):
        scl = scipy_zoom(scl, (Ht/scl.shape[0], Wt/scl.shape[1]), order=0)
    return np.isin(scl, [3, 8, 9, 10])


def evaluate_perpatch_detailed(pred, ref, mask=None, ps=256, dr=10000.0):
    """Evaluate per non-overlapping patch, return list of {y, x, psnr, ssim}."""
    C, H, W = pred.shape
    results = []
    n_total, n_cloudy, n_empty, n_inf = 0, 0, 0, 0
    for y in range(0, H-ps+1, ps):
        for x in range(0, W-ps+1, ps):
            n_total += 1
            pp, rp = pred[:, y:y+ps, x:x+ps], ref[:, y:y+ps, x:x+ps]
            if mask is not None and mask[y:y+ps, x:x+ps].mean() > 0.5:
                n_cloudy += 1
                continue
            if rp.mean() < 1.0:
                n_empty += 1
                continue
            p = compute_psnr(pp, rp, dr)
            s = compute_ssim(pp, rp, dr)
            if not np.isinf(p):
                results.append({'y': y, 'x': x, 'psnr': p, 'ssim': s})
            else:
                n_inf += 1
    print(f'    {n_total} total, {n_cloudy} cloudy-masked, {n_empty} empty, '
          f'{n_inf} inf, {len(results)} valid', flush=True)
    return results


def summarise_patch_results(results):
    if not results:
        return {'psnr_mean': 0, 'psnr_std': 0, 'ssim_mean': 0, 'ssim_std': 0, 'n_patches': 0}
    pa = np.array([r['psnr'] for r in results])
    sa = np.array([r['ssim'] for r in results])
    return {
        'psnr_mean': float(np.mean(pa)), 'psnr_std': float(np.std(pa)),
        'ssim_mean': float(np.mean(sa)), 'ssim_std': float(np.std(sa)),
        'n_patches': len(pa),
    }


# ═════════════════════════════════════════════════════════════════════
# VISUALISATION
# ═════════════════════════════════════════════════════════════════════

def s2_to_rgb(img, bands=(2,1,0), clip_range=(0, 2500)):
    rgb = img[list(bands)].astype(np.float32)
    lo, hi = clip_range
    return ((np.clip(rgb, lo, hi) - lo) / (hi - lo) * 255).astype(np.uint8).transpose(1,2,0)


def save_visualisations(cloudy_paths, ref_path, output_nb, output_bl,
                        ref_cloud_mask, output_dir):
    """Full-district thumbnails: inputs + no-blend vs blend."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    def load_thumbnail(path, factor=4):
        d, _ = load_geotiff(path)
        return d[:, ::factor, ::factor]

    thumbs = [load_thumbnail(p) for p in cloudy_paths]
    ref_thumb = load_thumbnail(ref_path)
    nb_thumb = output_nb[:, ::4, ::4]
    bl_thumb = output_bl[:, ::4, ::4]
    labels = [LABELS[k] for k in CLOUDY_KEYS] + [LABELS[REF_KEY]]

    # ── Figure 1: Inputs ──
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    for ax, img, lab in zip(axes, thumbs + [ref_thumb], labels):
        ax.imshow(s2_to_rgb(img.astype(np.float32))); ax.set_title(lab); ax.axis('off')
    fig.suptitle('True Colour (B4-B3-B2) — downsampled 4x', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'input_rgb.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print('  Saved input_rgb.png', flush=True)

    # ── Figure 2: Results comparison (2x4) ──
    ref_f = ref_thumb.astype(np.float32)
    fig, axes = plt.subplots(2, 4, figsize=(28, 13))

    for i, k in enumerate(CLOUDY_KEYS):
        axes[0,i].imshow(s2_to_rgb(thumbs[i].astype(np.float32)))
        axes[0,i].set_title(LABELS[k]); axes[0,i].axis('off')
    axes[0,3].imshow(s2_to_rgb(ref_f))
    axes[0,3].set_title(LABELS[REF_KEY]); axes[0,3].axis('off')

    axes[1,0].imshow(s2_to_rgb(nb_thumb))
    axes[1,0].set_title('No-Blend Output', color='#CC6600', fontweight='bold')
    axes[1,0].axis('off')

    axes[1,1].imshow(s2_to_rgb(bl_thumb))
    axes[1,1].set_title('Blend Output', color='green', fontweight='bold')
    axes[1,1].axis('off')

    diff_nb = np.mean(np.abs(nb_thumb - ref_f), axis=0)
    im1 = axes[1,2].imshow(diff_nb, cmap='hot', vmin=0, vmax=1000)
    axes[1,2].set_title('Diff (No-Blend)'); axes[1,2].axis('off')
    plt.colorbar(im1, ax=axes[1,2], fraction=.046, pad=.04)

    diff_bl = np.mean(np.abs(bl_thumb - ref_f), axis=0)
    im2 = axes[1,3].imshow(diff_bl, cmap='hot', vmin=0, vmax=1000)
    axes[1,3].set_title('Diff (Blend)'); axes[1,3].axis('off')
    plt.colorbar(im2, ax=axes[1,3], fraction=.046, pad=.04)

    fig.suptitle(f'PMAA Cloud Removal — {DISTRICT} (downsampled 4x)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'results_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print('  Saved results_comparison.png', flush=True)
    del thumbs, ref_thumb, ref_f, nb_thumb, bl_thumb


def save_patch_showcase(results_nb, results_bl, output_nb, output_bl, ref_f,
                        cloudy_paths, ps, output_dir):
    """
    Pick 5 patches (worst, 25th, median, 75th, best by blend PSNR).
    Each row: Input1 | Input2 | Input3 | No-Blend | Blend | Reference | Diff(Blend)
    Annotated with per-patch metrics.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    nb_lut = {(r['y'], r['x']): r for r in results_nb}
    bl_lut = {(r['y'], r['x']): r for r in results_bl}
    common = sorted(set(nb_lut.keys()) & set(bl_lut.keys()))
    if len(common) < 5:
        print('  Skipping patch showcase: not enough common patches.', flush=True)
        return

    common_sorted = sorted(common, key=lambda c: bl_lut[c]['psnr'])
    n = len(common_sorted)
    idxs = [0, max(1, n//4), n//2, min(n-2, 3*n//4), n-1]
    row_labels = ['Worst', '25th pctl', 'Median', '75th pctl', 'Best']
    selected = [common_sorted[i] for i in idxs]

    nrows, ncols = len(selected), 7
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*3.2, nrows*3.2))
    col_titles = [LABELS[CLOUDY_KEYS[0]], LABELS[CLOUDY_KEYS[1]], LABELS[CLOUDY_KEYS[2]],
                  'No-Blend', 'Blend', 'Reference', 'Diff (Blend)']

    for row, (y, x) in enumerate(selected):
        nb_m, bl_m = nb_lut[(y, x)], bl_lut[(y, x)]

        # Cloudy inputs (windowed read from disk)
        for ci in range(3):
            try:
                inp = read_patch_from_file(cloudy_paths[ci], y, x, ps)
                axes[row, ci].imshow(s2_to_rgb(inp.astype(np.float32)))
            except Exception:
                axes[row, ci].text(0.5, 0.5, 'N/A', ha='center', va='center',
                                   transform=axes[row, ci].transAxes)
            axes[row, ci].axis('off')
            if row == 0:
                axes[row, ci].set_title(col_titles[ci], fontsize=9)

        # No-blend
        nb_p = output_nb[:, y:y+ps, x:x+ps]
        axes[row, 3].imshow(s2_to_rgb(nb_p))
        axes[row, 3].axis('off')
        if row == 0:
            axes[row, 3].set_title(col_titles[3], fontsize=9, color='#CC6600',
                                   fontweight='bold')

        # Blend
        bl_p = output_bl[:, y:y+ps, x:x+ps]
        axes[row, 4].imshow(s2_to_rgb(bl_p))
        axes[row, 4].axis('off')
        if row == 0:
            axes[row, 4].set_title(col_titles[4], fontsize=9, color='green',
                                   fontweight='bold')

        # Reference
        ref_p = ref_f[:, y:y+ps, x:x+ps]
        axes[row, 5].imshow(s2_to_rgb(ref_p))
        axes[row, 5].axis('off')
        if row == 0:
            axes[row, 5].set_title(col_titles[5], fontsize=9)

        # Diff
        diff = np.mean(np.abs(bl_p - ref_p), axis=0)
        axes[row, 6].imshow(diff, cmap='hot', vmin=0, vmax=1500)
        axes[row, 6].axis('off')
        if row == 0:
            axes[row, 6].set_title(col_titles[6], fontsize=9)

        # Row annotation
        ann = (f'{row_labels[row]}   '
               f'NB: {nb_m["psnr"]:.1f}dB/{nb_m["ssim"]:.3f}   '
               f'BL: {bl_m["psnr"]:.1f}dB/{bl_m["ssim"]:.3f}')
        axes[row, 0].set_ylabel(ann, fontsize=8, rotation=0,
                                labelpad=200, va='center', ha='left')

    fig.suptitle(f'Patch Showcase — {DISTRICT}  (256x256, sorted by Blend PSNR)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0.15, 0, 1, 0.96])
    plt.savefig(os.path.join(output_dir, 'patch_showcase.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print('  Saved patch_showcase.png', flush=True)


def save_blend_zoom(output_nb, output_bl, ref_f, output_dir):
    """512x512 centre crop: no-blend (with grid lines) vs blend vs reference."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    _, H, W = output_nb.shape
    cy, cx = H // 2, W // 2
    sz = 512
    y0, x0 = max(0, cy - sz//2), max(0, cx - sz//2)
    y1, x1 = min(H, y0 + sz), min(W, x0 + sz)

    nb_c = output_nb[:, y0:y1, x0:x1]
    bl_c = output_bl[:, y0:y1, x0:x1]
    rf_c = ref_f[:, y0:y1, x0:x1]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(s2_to_rgb(nb_c))
    axes[0].set_title('No-Blend (block artifacts)', color='#CC6600', fontweight='bold')
    axes[0].axis('off')
    # Draw 256-px grid
    for p in range(0, sz+1, 256):
        axes[0].axhline(y=p, color='red', lw=0.5, alpha=0.5)
        axes[0].axvline(x=p, color='red', lw=0.5, alpha=0.5)

    axes[1].imshow(s2_to_rgb(bl_c))
    axes[1].set_title('Overlap-Blend (smooth)', color='green', fontweight='bold')
    axes[1].axis('off')

    axes[2].imshow(s2_to_rgb(rf_c))
    axes[2].set_title('Reference')
    axes[2].axis('off')

    fig.suptitle(f'{DISTRICT} — Centre 512x512 Crop  '
                 f'(red lines = 256px patch boundaries)',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'blend_zoom.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print('  Saved blend_zoom.png', flush=True)


# ═════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description=f'PMAA Cloud Removal — {DISTRICT}')
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--repo_dir', required=True)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--batch_size', type=int, default=8)
    args = parser.parse_args()

    import torch, gc

    device = args.device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}', flush=True)
    if device == 'cuda':
        print(f'  GPU: {torch.cuda.get_device_name(0)}', flush=True)
        print(f'  VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB', flush=True)

    CONFIG['batch_size'] = args.batch_size
    sys.path.insert(0, args.repo_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    # ── 1. Merge tiles ──
    print('\n[1/7] Merging tiled exports...', flush=True)
    merge_gee_tiles(args.data_dir)

    # ── 2. Discover files ──
    print('\n[2/7] Discovering files...', flush=True)
    groups = discover_exports(args.data_dir)
    for k in sorted(groups):
        print(f'  {k}:', flush=True)
        for t, p in groups[k].items():
            print(f'    {t}: {os.path.basename(p)} ({os.path.getsize(p)/1e6:.1f} MB)', flush=True)
    for k in CLOUDY_KEYS + [REF_KEY]:
        assert k in groups, f'"{k}" not found! Available: {sorted(groups)}'
        assert 'bands' in groups[k], f'No bands file for "{k}"'

    # ── 3. Load cloudy data ──
    print('\n[3/7] Loading cloudy data (uint16)...', flush=True)
    cloudy_data, cloudy_meta = [], []
    for k in CLOUDY_KEYS:
        d, m = load_geotiff(groups[k]['bands'])
        cloudy_data.append(d); cloudy_meta.append(m)
        print(f'  {LABELS[k]}: {d.shape} {d.dtype} ({d.nbytes/1e6:.0f} MB)', flush=True)
    print(f'  Total: {sum(d.nbytes for d in cloudy_data)/1e6:.0f} MB', flush=True)

    # ── 4. Load model ──
    print('\n[4/7] Loading model...', flush=True)
    model_path = os.path.join(args.repo_dir, 'pretrained', CONFIG['model_filename'])
    assert os.path.exists(model_path), f'Weights not found: {model_path}'
    model = load_pmaa_model(model_path, CONFIG['hidden_channels'],
                            CONFIG['out_channels'], device)

    # ── 4a. Inference: No-Blend ──
    print('\n[4a] Inference — No-Blend (overlap=0)...', flush=True)
    t0_nb = time.time()
    output_nb = run_inference_streaming(model, cloudy_data, CONFIG, device,
                                        overlap_override=0, label='no-blend')
    elapsed_nb = time.time() - t0_nb
    print(f'  Done in {elapsed_nb:.1f}s  |  shape={output_nb.shape}  '
          f'range=[{output_nb.min():.0f}, {output_nb.max():.0f}]', flush=True)

    # ── 4b. Inference: Blend ──
    print('\n[4b] Inference — Blend (overlap={})...'.format(CONFIG['overlap']), flush=True)
    t0_bl = time.time()
    output_bl = run_inference_streaming(model, cloudy_data, CONFIG, device,
                                        overlap_override=CONFIG['overlap'], label='blend')
    elapsed_bl = time.time() - t0_bl
    print(f'  Done in {elapsed_bl:.1f}s  |  shape={output_bl.shape}  '
          f'range=[{output_bl.min():.0f}, {output_bl.max():.0f}]', flush=True)

    # Free model + cloudy
    del model, cloudy_data
    if device == 'cuda':
        torch.cuda.empty_cache()
    gc.collect()
    print('  Freed model + cloudy data.', flush=True)

    # ── Save GeoTIFFs ──
    print('\n  Saving GeoTIFFs...', flush=True)
    save_geotiff(os.path.join(args.output_dir, f'{DISTRICT_PREFIX}_pmaa_noblend.tif'),
                 output_nb.astype(np.float32), cloudy_meta[0])
    save_geotiff(os.path.join(args.output_dir, f'{DISTRICT_PREFIX}_pmaa_blend.tif'),
                 output_bl.astype(np.float32), cloudy_meta[0])
    m16 = cloudy_meta[0].copy(); m16['dtype'] = 'uint16'
    save_geotiff(os.path.join(args.output_dir, f'{DISTRICT_PREFIX}_pmaa_blend_uint16.tif'),
                 np.clip(output_bl, 0, 10000).astype(np.uint16), m16)

    # ── 5. Evaluate ──
    print('\n[5/7] Evaluating...', flush=True)
    ref_data, _ = load_geotiff(groups[REF_KEY]['bands'])
    ref_f = np.nan_to_num(ref_data.astype(np.float32), nan=0.0)
    del ref_data; gc.collect()
    print(f'  Reference: {ref_f.shape}  range=[{ref_f.min():.0f}, {ref_f.max():.0f}]', flush=True)

    ref_cloud_mask = None
    if 'scl' in groups[REF_KEY]:
        scl_d, _ = load_geotiff(groups[REF_KEY]['scl'])
        ref_cloud_mask = create_scl_cloud_mask(scl_d, (ref_f.shape[1], ref_f.shape[2]))
        del scl_d; gc.collect()
        print(f'  Ref cloud mask: {100*ref_cloud_mask.mean():.1f}% masked', flush=True)

    print('  Evaluating No-Blend...', flush=True)
    results_nb = evaluate_perpatch_detailed(output_nb, ref_f, ref_cloud_mask,
                                            CONFIG['patch_size'], CONFIG['norm_max'])
    pp_nb = summarise_patch_results(results_nb)

    print('  Evaluating Blend...', flush=True)
    results_bl = evaluate_perpatch_detailed(output_bl, ref_f, ref_cloud_mask,
                                            CONFIG['patch_size'], CONFIG['norm_max'])
    pp_bl = summarise_patch_results(results_bl)

    # Print comparison table
    print(flush=True)
    print('  +----------------------------------------------------+', flush=True)
    print(f'  | {"Metric":<16} {"No-Blend":>14} {"Blend":>14}   |', flush=True)
    print('  +----------------------------------------------------+', flush=True)
    print(f'  | {"PSNR (dB)":<16} {pp_nb["psnr_mean"]:>7.2f}+/-{pp_nb["psnr_std"]:<5.2f} '
          f'{pp_bl["psnr_mean"]:>7.2f}+/-{pp_bl["psnr_std"]:<5.2f} |', flush=True)
    print(f'  | {"SSIM":<16} {pp_nb["ssim_mean"]:>7.4f}+/-{pp_nb["ssim_std"]:<5.4f} '
          f'{pp_bl["ssim_mean"]:>7.4f}+/-{pp_bl["ssim_std"]:<5.4f} |', flush=True)
    print(f'  | {"Patches":<16} {pp_nb["n_patches"]:>14} {pp_bl["n_patches"]:>14}   |', flush=True)
    print(f'  | {"Time (s)":<16} {elapsed_nb:>14.1f} {elapsed_bl:>14.1f}   |', flush=True)
    print('  +----------------------------------------------------+', flush=True)

    # Save metrics
    with open(os.path.join(args.output_dir, 'metrics.txt'), 'w') as f:
        f.write(f'PMAA Cloud Removal  —  {DISTRICT} District\n')
        f.write('=' * 58 + '\n\n')
        f.write(f'{"Metric":<20} {"No-Blend":>16} {"Blend":>16}\n')
        f.write('-' * 58 + '\n')
        f.write(f'{"PSNR (dB)":<20} '
                f'{pp_nb["psnr_mean"]:.3f}+/-{pp_nb["psnr_std"]:.3f}    '
                f'{pp_bl["psnr_mean"]:.3f}+/-{pp_bl["psnr_std"]:.3f}\n')
        f.write(f'{"SSIM":<20} '
                f'{pp_nb["ssim_mean"]:.4f}+/-{pp_nb["ssim_std"]:.4f}   '
                f'{pp_bl["ssim_mean"]:.4f}+/-{pp_bl["ssim_std"]:.4f}\n')
        f.write(f'{"Patches":<20} {pp_nb["n_patches"]:>16} {pp_bl["n_patches"]:>16}\n')
        f.write('-' * 58 + '\n')
        f.write(f'{"Inference time":<20} {elapsed_nb:>15.1f}s {elapsed_bl:>15.1f}s\n')
        f.write(f'{"Overlap (px)":<20} {"0":>16} {CONFIG["overlap"]:>16}\n')
        f.write(f'{"Patch size":<20} {CONFIG["patch_size"]:>16}\n')
        f.write(f'\nDevice: {device}\n')
        f.write(f'Output shape: {output_bl.shape}\n')

    # ── 6. Visualisations ──
    print('\n[6/7] Patch showcase + blend zoom...', flush=True)
    cloudy_paths = [groups[k]['bands'] for k in CLOUDY_KEYS]

    save_patch_showcase(results_nb, results_bl,
                        output_nb, output_bl, ref_f,
                        cloudy_paths, CONFIG['patch_size'], args.output_dir)

    save_blend_zoom(output_nb, output_bl, ref_f, args.output_dir)

    del ref_f; gc.collect()

    # ── 7. Full-district overview ──
    print('\n[7/7] Full-district visualisations...', flush=True)
    ref_path = groups[REF_KEY]['bands']
    save_visualisations(cloudy_paths, ref_path, output_nb, output_bl,
                        ref_cloud_mask, args.output_dir)

    # ── Done ──
    print('\n' + '=' * 55, flush=True)
    print('  DONE — Results:', flush=True)
    for fname in sorted(os.listdir(args.output_dir)):
        sz = os.path.getsize(os.path.join(args.output_dir, fname)) / 1e6
        print(f'    {fname} ({sz:.1f} MB)', flush=True)
    print('=' * 55, flush=True)


if __name__ == '__main__':
    main()
