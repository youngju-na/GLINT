#!/usr/bin/env python3
"""
BlenderProc-3DFront to EasyVolCap Format Converter

This script converts BlenderProc-3DFront rendering outputs (HDF5 files) to EasyVolCap format.
Supports RGB images, depth maps, normal maps, masks, and camera parameters.

Usage:
    python blenderproc_to_easyvolcap.py --render_root ./renderings --output_root ./dataset_easyvolcap

Author: BlenderProc-3DFront Team
"""

import os
import cv2
import json
import argparse
import h5py
import numpy as np
from pathlib import Path
from PIL import Image

try:
    from easyvolcap.utils.console_utils import *
    from easyvolcap.utils.base_utils import dotdict
    from easyvolcap.utils.math_utils import normalize
    from easyvolcap.utils.easy_utils import write_camera
    from easyvolcap.utils.data_utils import save_image, load_image
    from easyvolcap.utils.parallel_utils import parallel_execution
    EASYVOLCAP_AVAILABLE = True
except ImportError:
    print("Warning: EasyVolCap not found. Some features may not work.")
    EASYVOLCAP_AVAILABLE = False
    
    # Fallback implementations
    class dotdict(dict):
        def __getattr__(self, key):
            return self[key]
        def __setattr__(self, key, value):
            self[key] = value
    
    def save_image(path, img):
        if img.max() <= 1.01:
            img = (img * 255.0).astype(np.uint8)
        Image.fromarray(img).save(path)
    
    def write_camera(cameras, output_dir):
        # Simple fallback camera writer
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        intri_data = {}
        extri_data = {}
        
        for cam_name, cam_params in cameras.items():
            intri_data[cam_name] = {
                'K': cam_params['K'].tolist(),
                'D': cam_params['D'].tolist(),
                'H': cam_params['H'],
                'W': cam_params['W']
            }
            extri_data[cam_name] = {
                'R': cam_params['R'].tolist(),
                'T': cam_params['T'].tolist()
            }
        
        with open(output_dir / 'intri.json', 'w') as f:
            json.dump(intri_data, f, indent=2)
        with open(output_dir / 'extri.json', 'w') as f:
            json.dump(extri_data, f, indent=2)


def convert_to_easyvolcap_format(render_dir, easyvolcap_root, has_alpha=False, has_normal=True, has_depth=True,
                                black_bkgd=False, ext='png', cam_K_path=None, verbose=True):
    """
    Convert BlenderProc-3DFront rendering output to EasyVolCap format
    
    Args:
        render_dir: Path to rendered HDF5 files
        easyvolcap_root: Output path for EasyVolCap format
        has_alpha: Whether to save alpha channel
        has_normal: Whether to save normal maps
        has_depth: Whether to save depth maps
        black_bkgd: Use black background (True) or white (False)
        ext: Image file extension
        cam_K_path: Path to camera intrinsic matrix file
        verbose: Print progress information
    
    Returns:
        Number of cameras converted
    """
    render_dir = Path(render_dir)
    easyvolcap_root = Path(easyvolcap_root)
    
    if verbose:
        print(f"Converting {render_dir} to EasyVolCap format at {easyvolcap_root}")
    
    # Background color
    bg_color = np.array([0, 0, 0]) if black_bkgd else np.array([1, 1, 1])
    
    # Create output directories for train split
    split = 'train'
    img_out_dir = easyvolcap_root / split / 'images'
    msk_out_dir = easyvolcap_root / split / 'masks'
    if has_alpha: 
        alpha_out_dir = easyvolcap_root / split / 'alphas'
    if has_normal: 
        normal_out_dir = easyvolcap_root / split / 'normals_gt'
    if has_depth:
        depth_out_dir = easyvolcap_root / split / 'depths'
    
    img_out_dir.mkdir(parents=True, exist_ok=True)
    msk_out_dir.mkdir(parents=True, exist_ok=True)
    if has_alpha: 
        alpha_out_dir.mkdir(parents=True, exist_ok=True)
    if has_normal: 
        normal_out_dir.mkdir(parents=True, exist_ok=True)
    if has_depth:
        depth_out_dir.mkdir(parents=True, exist_ok=True)
    
    # Find HDF5 files
    h5_paths = sorted(list(render_dir.glob('*.h5')) + list(render_dir.glob('*.hdf5')))
    if len(h5_paths) == 0:
        print(f'No .h5/.hdf5 files found in {render_dir}')
        return 0
    
    if verbose:
        print(f"Found {len(h5_paths)} HDF5 files")
    
    # Load camera intrinsics if available
    K_matrix = None
    if cam_K_path and Path(cam_K_path).exists():
        try:
            K_matrix = np.load(cam_K_path)
            if verbose:
                print(f"Loaded camera intrinsics from {cam_K_path}")
        except Exception as e:
            print(f'Failed to load camera intrinsics: {e}')
    
    cameras = dotdict()
    
    for cnt, h5_path in enumerate(h5_paths):
        if verbose and cnt % 10 == 0:
            print(f"Processing file {cnt+1}/{len(h5_paths)}: {h5_path.name}")
            
        with h5py.File(h5_path, 'r') as f:
            color = np.array(f['colors']) if 'colors' in f else None
            depth = np.array(f['depth']) if 'depth' in f else None
            normals = np.array(f['normals']) if 'normals' in f else None
            cam_Ts = np.array(f['cam_Ts']) if 'cam_Ts' in f else None
        
        if color is None:
            print(f'No color data in {h5_path}')
            continue
            
        # Handle color data format
        if color.ndim == 4:  # Multiple views in one file
            n_views = color.shape[0]
        else:  # Single view
            color = color[np.newaxis, ...]
            n_views = 1
            
        # Handle camera transforms
        if cam_Ts is not None:
            if cam_Ts.ndim == 2:  # Single camera
                cam_Ts = cam_Ts[np.newaxis, ...]
        else:
            print(f'No camera transforms in {h5_path}')
            continue
            
        # Process each view
        for view_idx in range(n_views):
            frame_idx = cnt * n_views + view_idx
            
            # Get current view data
            curr_color = color[view_idx]
            curr_cam_T = cam_Ts[view_idx] if view_idx < len(cam_Ts) else cam_Ts[0]
            if depth is not None:
                curr_depth = depth[view_idx] if depth.ndim >= 3 else depth
            
            # Handle color format (C,H,W) -> (H,W,C)
            if curr_color.ndim == 3 and curr_color.shape[0] in (3, 4) and curr_color.shape[2] not in (3, 4):
                curr_color = np.transpose(curr_color, (1, 2, 0))
            
            H, W = curr_color.shape[:2]
            
            # Process RGB image
            if curr_color.shape[-1] == 4:  # RGBA
                rgb = curr_color[:, :, :3] * curr_color[:, :, 3:] + bg_color * (1 - curr_color[:, :, 3:])
                alpha = curr_color[:, :, 3]
            else:  # RGB
                rgb = curr_color[:, :, :3]
                alpha = np.ones((H, W), dtype=np.float32)
            
            # Ensure RGB is in [0,1] range
            if rgb.max() > 1.01:
                rgb = rgb / 255.0
            
            # Save RGB image
            img_easyvolcap_path = img_out_dir / f'{frame_idx:04d}' / f'000000.{ext}'
            img_easyvolcap_path.parent.mkdir(exist_ok=True)
            save_image(str(img_easyvolcap_path), rgb)
            
            # Create and save mask
            mask = (alpha > 0.5).astype(np.uint8) * 255
            msk_easyvolcap_path = msk_out_dir / f'{frame_idx:04d}' / f'000000.{ext}'
            msk_easyvolcap_path.parent.mkdir(exist_ok=True)
            cv2.imwrite(str(msk_easyvolcap_path), mask)
            
            # Save alpha if requested
            if has_alpha:
                alpha_easyvolcap_path = alpha_out_dir / f'{frame_idx:04d}' / f'000000.{ext}'
                alpha_easyvolcap_path.parent.mkdir(exist_ok=True)
                alpha_img = (alpha * 255).astype(np.uint8)
                cv2.imwrite(str(alpha_easyvolcap_path), alpha_img)
            
            # Save normals if available and requested
            if has_normal and normals is not None:
                curr_normals = normals[view_idx] if normals.ndim == 4 else normals
                if curr_normals.ndim == 3 and curr_normals.shape[0] == 3:
                    curr_normals = np.transpose(curr_normals, (1, 2, 0))
                
                # Convert normals from [-1,1] to [0,255]
                normal_img = ((curr_normals * 0.5 + 0.5) * 255).astype(np.uint8)
                normal_easyvolcap_path = normal_out_dir / f'{frame_idx:04d}' / f'000000.{ext}'
                normal_easyvolcap_path.parent.mkdir(exist_ok=True)
                cv2.imwrite(str(normal_easyvolcap_path), normal_img)
            
            # Save depth if available and requested
            if has_depth and depth is not None:
                # Handle depth format - ensure it's 2D
                if curr_depth.ndim == 3 and curr_depth.shape[-1] == 1:
                    curr_depth = curr_depth[:, :, 0]
                elif curr_depth.ndim == 3 and curr_depth.shape[0] == 1:
                    curr_depth = curr_depth[0, :, :]
                
                # Save raw depth as .npy for precise values
                depth_npy_path = depth_out_dir / f'{frame_idx:04d}' / f'000000.npy'
                depth_npy_path.parent.mkdir(exist_ok=True)
                np.save(str(depth_npy_path), curr_depth.astype(np.float32))
                
                # Save depth as 16-bit PNG (scaled to millimeters)
                depth_valid = np.isfinite(curr_depth) & (curr_depth > 0)
                depth_scaled = np.zeros_like(curr_depth, dtype=np.uint16)
                if depth_valid.any():
                    # Convert to millimeters and clip to 16-bit range
                    depth_mm = curr_depth * 1000.0  # meters to millimeters
                    depth_scaled[depth_valid] = np.clip(depth_mm[depth_valid], 0, 65535).astype(np.uint16)
                
                depth_png_path = depth_out_dir / f'{frame_idx:04d}' / f'000000.{ext}'
                Image.fromarray(depth_scaled).save(str(depth_png_path))
            
            # Process camera parameters
            if curr_cam_T.shape == (4, 4):
                # BlenderProc uses OpenGL convention, convert to OpenCV
                c2w_opengl = curr_cam_T.astype(np.float32)
                c2w_opencv = c2w_opengl @ np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
                w2c_opencv = np.linalg.inv(c2w_opencv)
                
                # Set up camera parameters
                if K_matrix is not None:
                    K = K_matrix.copy()
                else:
                    # Use default intrinsics if not available
                    focal = W * 0.7  # Rough estimate
                    K = np.array([[focal, 0, W/2],
                                  [0, focal, H/2], 
                                  [0, 0, 1]], dtype=np.float32)
                
                cameras[f'{frame_idx:04d}'] = {
                    'R': w2c_opencv[:3, :3],
                    'T': w2c_opencv[:3, 3:],
                    'K': K,
                    'D': np.zeros((1, 5)),
                    'H': H, 'W': W,
                }
    
    # Write camera parameters
    write_camera(cameras, str(easyvolcap_root / split))
    
    if verbose:
        print(f'Converted {len(cameras)} cameras saved to {easyvolcap_root / split}')
        
    return len(cameras)


def process_all_scenes_to_easyvolcap(render_root, easyvolcap_root, **kwargs):
    """
    Process all scenes in render_root to EasyVolCap format
    
    Args:
        render_root: Root directory containing scene folders with HDF5 files
        easyvolcap_root: Output directory for EasyVolCap format
        **kwargs: Additional arguments for convert_to_easyvolcap_format
    """
    render_root = Path(render_root)
    easyvolcap_root = Path(easyvolcap_root)
    
    if not render_root.exists():
        print(f"Error: Render root directory does not exist: {render_root}")
        return
    
    scenes = [d for d in render_root.iterdir() if d.is_dir()]
    scenes = sorted(scenes)
    
    if len(scenes) == 0:
        print(f"No scene directories found in {render_root}")
        return
    
    print(f"Found {len(scenes)} scenes to process")
    
    successful_conversions = 0
    total_cameras = 0
    
    for scene_dir in scenes:
        scene_name = scene_dir.name
        print(f'\n=== Processing scene: {scene_name} ===')
        
        # Check for cam_K.npy in scene directory
        cam_K_path = scene_dir / 'cam_K.npy'
        if not cam_K_path.exists():
            cam_K_path = scene_dir.parent / 'cam_K.npy'  # Check parent directory
            if not cam_K_path.exists():
                cam_K_path = None
        
        try:
            n_cams = convert_to_easyvolcap_format(
                render_dir=scene_dir,
                easyvolcap_root=easyvolcap_root / scene_name,
                cam_K_path=str(cam_K_path) if cam_K_path else None,
                **kwargs
            )
            print(f'✓ Successfully converted {n_cams} cameras for scene {scene_name}')
            successful_conversions += 1
            total_cameras += n_cams
            
        except Exception as e:
            print(f'✗ Failed to process scene {scene_name}: {e}')
    
    print(f'\n=== Conversion Summary ===')
    print(f'Successfully processed: {successful_conversions}/{len(scenes)} scenes')
    print(f'Total cameras converted: {total_cameras}')
    print(f'Output directory: {easyvolcap_root}')


def main():
    parser = argparse.ArgumentParser(description='Convert BlenderProc-3DFront output to EasyVolCap format')
    parser.add_argument('--render_root', type=str, required=True,
                        help='Root directory containing scene folders with HDF5 files')
    parser.add_argument('--output_root', type=str, required=True,
                        help='Output directory for EasyVolCap format')
    parser.add_argument('--has_alpha', action='store_true',
                        help='Save alpha channel')
    parser.add_argument('--has_normal', action='store_true', default=True,
                        help='Save normal maps (default: True)')
    parser.add_argument('--has_depth', action='store_true', default=True,
                        help='Save depth maps (default: True)')
    parser.add_argument('--black_bkgd', action='store_true',
                        help='Use black background instead of white')
    parser.add_argument('--ext', type=str, default='png',
                        help='Image file extension (default: png)')
    parser.add_argument('--single_scene', type=str, default=None,
                        help='Process only a single scene by name')
    parser.add_argument('--verbose', action='store_true', default=True,
                        help='Print verbose output')
    
    args = parser.parse_args()
    
    if not EASYVOLCAP_AVAILABLE:
        print("Warning: EasyVolCap library not available. Using fallback implementations.")
    
    if args.single_scene:
        # Process single scene
        scene_path = Path(args.render_root) / args.single_scene
        if not scene_path.exists():
            print(f"Error: Scene directory does not exist: {scene_path}")
            return
        
        cam_K_path = scene_path / 'cam_K.npy'
        if not cam_K_path.exists():
            cam_K_path = scene_path.parent / 'cam_K.npy'
            if not cam_K_path.exists():
                cam_K_path = None
        
        n_cams = convert_to_easyvolcap_format(
            render_dir=scene_path,
            easyvolcap_root=Path(args.output_root) / args.single_scene,
            has_alpha=args.has_alpha,
            has_normal=args.has_normal,
            has_depth=args.has_depth,
            black_bkgd=args.black_bkgd,
            ext=args.ext,
            cam_K_path=str(cam_K_path) if cam_K_path else None,
            verbose=args.verbose
        )
        print(f"Converted {n_cams} cameras for scene {args.single_scene}")
    else:
        # Process all scenes
        process_all_scenes_to_easyvolcap(
            render_root=args.render_root,
            easyvolcap_root=args.output_root,
            has_alpha=args.has_alpha,
            has_normal=args.has_normal,
            has_depth=args.has_depth,
            black_bkgd=args.black_bkgd,
            ext=args.ext,
            verbose=args.verbose
        )


if __name__ == '__main__':
    main()
