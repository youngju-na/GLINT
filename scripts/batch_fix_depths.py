#!/usr/bin/env python3
"""
Batch Fix GT Depths Script

Permanently fixes GT depth maps for Scenes 4-8 by:
1. Filtering outliers (depth > 100m -> 0)
2. Correcting spherical distortion (depth = depth / rz)

This standardizes the dataset to valid Planar Z depth.
"""

import os
import glob
import argparse
import numpy as np
from tqdm import tqdm
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def read_cameras_txt(path):
    """
    Minimal parser for COLMAP cameras.txt to get K
    """
    cameras = {}
    with open(path, "r") as f:
        for line in f:
            if line.startswith("#"): continue
            parts = line.strip().split()
            cam_id = int(parts[0])
            model = parts[1]
            W = int(parts[2])
            H = int(parts[3])
            params = [float(x) for x in parts[4:]]
            
            if model == "PINHOLE":
                fx, fy, cx, cy = params[:4]
            elif model == "SIMPLE_PINHOLE":
                fx = fy = params[0]
                cx, cy = params[1], params[2]
            elif model in ["SIMPLE_RADIAL", "RADIAL"]:
                fx = fy = params[0]
                cx, cy = params[1], params[2]
            elif model == "OPENCV":
                fx, fy, cx, cy = params[:4]
            else:
                print(f"[WARN] Unknown camera model {model}, assuming PINHOLE-like first 4 params")
                fx, fy, cx, cy = params[:4]
                
            K = np.array([
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1]
            ])
            cameras[cam_id] = (W, H, K)
    return cameras

def correct_depth_distortion(depth, K):
    from numpy import sqrt
    H, W = depth.shape
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    
    u = np.arange(W, dtype=np.float32)
    v = np.arange(H, dtype=np.float32)
    uu, vv = np.meshgrid(u, v)
    
    d_x = (uu - cx) / fx
    d_y = (vv - cy) / fy
    d_z = np.ones_like(d_x)
    
    # rz = d_z / norm(d)
    norm = sqrt(d_x**2 + d_y**2 + d_z**2)
    rz = d_z / norm
    rz = np.maximum(rz, 1e-6)
    
    return depth / rz

def process_scene(scene_path):
    scene_name = os.path.basename(scene_path)
    print(f"\nProcessing {scene_name}...")
    
    # 1. Find cameras.txt
    cam_path = os.path.join(scene_path, "sparse/0/cameras.txt")
    if not os.path.exists(cam_path):
        cam_path = os.path.join(scene_path, "cameras.txt")
        
    if not os.path.exists(cam_path):
        print(f"[ERR] Could not find cameras.txt in {scene_path}")
        return
        
    print(f"  Loaded cameras from {cam_path}")
    cameras = read_cameras_txt(cam_path)
    if not cameras:
        print("[ERR] No cameras found")
        return
        
    # Assume single camera or use first one for K (typical for synthetic structure)
    W, H, K = list(cameras.values())[0]
    print(f"  Camera: {W}x{H}, fx={K[0,0]:.2f}, fy={K[1,1]:.2f}")
    
    # 2. Find depth files
    depth_dir = os.path.join(scene_path, "depths_gt")
    # Check if depths are in root (flat structure case)
    if not os.path.isdir(depth_dir):
        depth_dir = scene_path
        
    depth_files = sorted(glob.glob(os.path.join(depth_dir, "val_depthCamZ_*.npy")))
    if not depth_files:
        print(f"[ERR] No depth files found in {depth_dir}")
        return
        
    print(f"  Found {len(depth_files)} depth files")
    
    # 3. Create output directory
    output_dir = os.path.join(scene_path, "depths_gt_fixed")
    os.makedirs(output_dir, exist_ok=True)
    print(f"  Saving to {output_dir}")
    
    # 4. Process loop
    for fpath in tqdm(depth_files, desc=f"  Fixing {scene_name}", unit="frame"):
        depth = np.load(fpath)
        
        # Validation cap (100m)
        mask = depth > 100.0
        if mask.any():
            depth[mask] = 0.0
            
        # Correction (Un-distort)
        depth_fixed = correct_depth_distortion(depth, K)
        
        # Save
        fname = os.path.basename(fpath)
        save_path = os.path.join(output_dir, fname)
        np.save(save_path, depth_fixed)

def main():
    base_root = "data/datasets"
    scenes = [4, 5, 6, 7, 8, "5_redo"] # Added 5_redo just in case
    
    for s in scenes:
        scene_dir = os.path.join(base_root, f"scene_{s}")
        if os.path.exists(scene_dir):
            process_scene(scene_dir)
        else:
            print(f"[WARN] Scene directory not found: {scene_dir}")

if __name__ == "__main__":
    main()
