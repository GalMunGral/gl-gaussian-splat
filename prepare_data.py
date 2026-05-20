"""
Download the truck Gaussian Splat PLY from HuggingFace and export public/truck.bin.

Requirements:
    pip install huggingface_hub plyfile numpy

Usage:
    python prepare_data.py
"""

import os
import numpy as np
from plyfile import PlyData
from huggingface_hub import hf_hub_download

OPACITY_THRESHOLD = 0.1
SIZE_THRESHOLD    = 0.01   # world-space units; max extent across all 3 principal axes

def build_cov(scale, quat):
    s = np.exp(scale)
    w, x, y, z = quat[:,0], quat[:,1], quat[:,2], quat[:,3]

    R = np.stack([
        1-2*(y*y+z*z),  2*(x*y-w*z),   2*(x*z+w*y),
        2*(x*y+w*z),    1-2*(x*x+z*z), 2*(y*z-w*x),
        2*(x*z-w*y),    2*(y*z+w*x),   1-2*(x*x+y*y),
    ], axis=1).reshape(-1, 3, 3)

    S2 = np.zeros((len(scale), 3, 3))
    S2[:,0,0] = s[:,0]**2
    S2[:,1,1] = s[:,1]**2
    S2[:,2,2] = s[:,2]**2

    cov = R @ S2 @ R.transpose(0, 2, 1)
    return np.stack([
        cov[:,0,0], cov[:,0,1], cov[:,0,2],
        cov[:,1,1], cov[:,1,2], cov[:,2,2],
    ], axis=1)


def main():
    print("Downloading PLY from HuggingFace...")
    ply_path = hf_hub_download(
        repo_id   = "Voxel51/gaussian_splatting",
        filename  = "FO_dataset/truck/point_cloud/iteration_30000/point_cloud.ply",
        repo_type = "dataset",
        local_dir = "data",
    )
    print(f"  -> {ply_path}")

    ply = PlyData.read(ply_path)
    v   = ply['vertex']
    print(f"  {len(v)} Gaussians loaded")

    # position (flip Y to match WebGPU convention)
    xyz = np.stack([v['x'], -v['y'], v['z']], axis=1).astype(np.float32)

    # SH coefficients — DC + 15 rest terms per channel, interleaved R/G/B in the PLY
    f_dc_raw = np.stack([v['f_dc_0'], v['f_dc_1'], v['f_dc_2']], axis=1).astype(np.float32)
    f_rest_R = np.stack([v[f'f_rest_{i}'] for i in range( 0, 15)], axis=1).astype(np.float32)
    f_rest_G = np.stack([v[f'f_rest_{i}'] for i in range(15, 30)], axis=1).astype(np.float32)
    f_rest_B = np.stack([v[f'f_rest_{i}'] for i in range(30, 45)], axis=1).astype(np.float32)

    # sigmoid opacity + world-space covariance
    opacity = (1.0 / (1.0 + np.exp(-v['opacity'].astype(np.float32))))
    scale   = np.stack([v['scale_0'], v['scale_1'], v['scale_2']], axis=1).astype(np.float32)
    quat    = np.stack([v['rot_0'],   v['rot_1'],   v['rot_2'],   v['rot_3']], axis=1).astype(np.float32)
    quat   /= np.linalg.norm(quat, axis=1, keepdims=True)
    cov     = build_cov(scale, quat)

    # filter
    max_scale = np.exp(scale).max(axis=1)
    mask = (opacity > OPACITY_THRESHOLD) & (max_scale > SIZE_THRESHOLD)
    print(f"  kept {mask.sum()} / {len(v)} Gaussians ({100*mask.mean():.1f}%)")

    pad  = np.zeros((mask.sum(), 6), dtype=np.float32)
    data = np.concatenate([
        xyz[mask],                          # 0-2
        f_dc_raw[mask],                     # 3-5
        f_rest_R[mask],                     # 6-20
        f_rest_G[mask],                     # 21-35
        f_rest_B[mask],                     # 36-50
        opacity[mask].reshape(-1, 1),       # 51
        cov[mask],                          # 52-57
        pad,                                # 58-63
    ], axis=1).astype(np.float32)
    assert data.shape[1] == 64, f"expected 64 floats per Gaussian, got {data.shape[1]}"

    os.makedirs('public', exist_ok=True)
    out = 'public/truck.bin'
    data.tofile(out)
    print(f"  written: {out}  ({data.nbytes / 1e6:.1f} MB)  —  {mask.sum()} Gaussians x 256 bytes")


if __name__ == '__main__':
    main()