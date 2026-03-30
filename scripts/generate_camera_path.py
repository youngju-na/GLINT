"""
Generate smooth camera paths between two camera indices for project page videos.

Supports multiple path types commonly used in 3DGS project pages:
  - arc:    Smooth arc around the scene's center of attention (most common for object-centric scenes)
  - slerp:  Spherical linear interpolation for rotation + linear for position
  - spiral: Local spiral segment between two cameras

All paths apply cosine ease-in-out for cinematic smoothness.

Usage:
    python scripts/generate_camera_path.py \
        --data_root data/datasets/ref-dl3dv/scene_name \
        --cam_idx1 0 --cam_idx2 16 \
        --n_frames 120 --path_type arc \
        --output data/camera_paths/scene_name_0_16
"""

import os
import cv2
import argparse
import numpy as np
from os.path import join, exists
from scipy.spatial.transform import Rotation, Slerp

# ────────────────────────── helpers ──────────────────────────

def normalize(v):
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-13)


def cosine_ease(t):
    """Cosine ease-in-out: slow at endpoints, fast in the middle."""
    return 0.5 * (1 - np.cos(np.pi * t))


def look_at(eye, target, up):
    """Construct a c2w matrix (3x4) following EasyVolcap convention.
    Columns: [right, down, forward, center]
    Matches cam_utils.py generate_spiral_path convention."""
    v_front = normalize(target - eye)                  # forward: eye -> target
    v_right = normalize(np.cross(v_front, up))         # right
    v_down = np.cross(v_front, v_right)                # down
    return np.stack([v_right, v_down, v_front, eye], axis=-1).astype(np.float32)  # 3x4


def compute_focus_point(c2w1, c2w2):
    """Approximate focus point as closest point between the two optical axes."""
    o1, d1 = c2w1[:3, 3], c2w1[:3, 2]
    o2, d2 = c2w2[:3, 3], c2w2[:3, 2]

    # Closest point between two rays
    d1, d2 = normalize(d1), normalize(d2)
    cross = np.cross(d1, d2)
    denom = np.dot(cross, cross) + 1e-10
    t = o2 - o1
    t1 = np.linalg.det(np.stack([t, d2, cross])) / denom
    t2 = np.linalg.det(np.stack([t, d1, cross])) / denom
    p1 = o1 + t1 * d1
    p2 = o2 + t2 * d2
    return 0.5 * (p1 + p2)


# ────────────────────────── path generators ──────────────────────────

def generate_arc_path(c2w1, c2w2, n_frames):
    """
    Arc path: cameras move along a circular arc while always looking at the
    scene's center of attention.  This is the standard "turntable segment"
    used in most 3DGS project page videos.
    """
    # Derive up vector from cameras (EasyVolcap: column 1 is down, so up = -column1)
    up = -normalize(0.5 * (c2w1[:3, 1] + c2w2[:3, 1]))

    focus = compute_focus_point(c2w1, c2w2)

    pos1, pos2 = c2w1[:3, 3], c2w2[:3, 3]

    # Vectors from focus to each camera position
    v1 = pos1 - focus
    v2 = pos2 - focus
    r1, r2 = np.linalg.norm(v1), np.linalg.norm(v2)

    # Build an orthonormal frame on the arc plane
    e1 = normalize(v1)
    # Component of v2 orthogonal to e1
    v2_orth = v2 - np.dot(v2, e1) * e1
    e2 = normalize(v2_orth)

    # Angle between the two radial directions
    cos_angle = np.clip(np.dot(normalize(v1), normalize(v2)), -1, 1)
    total_angle = np.arccos(cos_angle)

    ts = cosine_ease(np.linspace(0, 1, n_frames, dtype=np.float32))

    R1, R2 = c2w1[:3, :3], c2w2[:3, :3]
    rots = Rotation.from_matrix(np.stack([R1, R2]))
    slerp = Slerp([0.0, 1.0], rots)

    c2ws = []
    for t in ts:
        angle = total_angle * t
        r = r1 + (r2 - r1) * t  # smoothly interpolate radius
        pos = focus + r * (np.cos(angle) * e1 + np.sin(angle) * e2)
        R = slerp(float(t)).as_matrix().astype(np.float32)
        c2ws.append(np.concatenate([R, pos[:, None].astype(np.float32)], axis=-1))

    return np.stack(c2ws, axis=0)  # (N, 3, 4)


def generate_slerp_path(c2w1, c2w2, n_frames):
    """
    Slerp path: spherical interpolation for rotation, linear for position,
    both with cosine ease-in-out.
    """
    pos1, pos2 = c2w1[:3, 3], c2w2[:3, 3]
    R1, R2 = c2w1[:3, :3], c2w2[:3, :3]

    rots = Rotation.from_matrix(np.stack([R1, R2]))
    slerp = Slerp([0.0, 1.0], rots)

    ts = cosine_ease(np.linspace(0, 1, n_frames, dtype=np.float32))

    c2ws = []
    for t in ts:
        R = slerp(t).as_matrix().astype(np.float32)
        pos = ((1 - t) * pos1 + t * pos2).astype(np.float32)
        c2ws.append(np.concatenate([R, pos[:, None]], axis=-1))

    return np.stack(c2ws, axis=0)


def generate_spiral_segment_path(c2w1, c2w2, n_frames, n_rots=0.5, amplitude=0.15):
    """
    Spiral segment: follows a slerp baseline but adds a gentle spiral wobble.
    Good for showcasing view-dependent effects.
    """
    # Derive up vector from cameras (EasyVolcap: column 1 is down, so up = -column1)
    up = -normalize(0.5 * (c2w1[:3, 1] + c2w2[:3, 1]))

    focus = compute_focus_point(c2w1, c2w2)
    pos1, pos2 = c2w1[:3, 3], c2w2[:3, 3]
    R1, R2 = c2w1[:3, :3], c2w2[:3, :3]

    # Baseline: the right-vector from camera 1 for spiral offset direction
    v_front = normalize(focus - pos1)
    right = normalize(np.cross(v_front, up))
    up_dir = normalize(np.cross(v_front, right))

    rots = Rotation.from_matrix(np.stack([R1, R2]))
    slerp = Slerp([0.0, 1.0], rots)

    ts_raw = np.linspace(0, 1, n_frames, dtype=np.float32)
    ts = cosine_ease(ts_raw)

    baseline_dist = np.linalg.norm(pos2 - pos1)
    spiral_r = amplitude * baseline_dist

    c2ws = []
    for t_raw, t in zip(ts_raw, ts):
        base_pos = (1 - t) * pos1 + t * pos2
        # Add spiral offset
        theta = 2 * np.pi * n_rots * t_raw
        offset = spiral_r * (np.cos(theta) * right + np.sin(theta) * up_dir)
        # Attenuate at endpoints for smooth start/end
        envelope = np.sin(np.pi * t_raw)
        pos = base_pos + offset * envelope
        c2ws.append(look_at(pos.astype(np.float32), focus, up))

    return np.stack(c2ws, axis=0)


# ────────────────────────── camera I/O (EasyVolcap format) ──────────────────────────

def load_cameras_from_easyvolcap(data_root):
    """Load cameras from EasyVolcap intri.yml / extri.yml."""
    intri_path = join(data_root, 'intri.yml')
    extri_path = join(data_root, 'extri.yml')
    assert exists(intri_path), f'intri.yml not found at {intri_path}'
    assert exists(extri_path), f'extri.yml not found at {extri_path}'

    intri_fs = cv2.FileStorage(intri_path, cv2.FILE_STORAGE_READ)
    extri_fs = cv2.FileStorage(extri_path, cv2.FILE_STORAGE_READ)

    names = []
    names_node = intri_fs.getNode('names')
    for i in range(names_node.size()):
        names.append(names_node.at(i).string())

    cameras = {}
    for name in names:
        cam = {}
        cam['K'] = intri_fs.getNode(f'K_{name}').mat()
        cam['H'] = int(intri_fs.getNode(f'H_{name}').real()) or -1
        cam['W'] = int(intri_fs.getNode(f'W_{name}').real()) or -1

        Rvec = extri_fs.getNode(f'R_{name}').mat()
        Tvec = extri_fs.getNode(f'T_{name}').mat()
        if Rvec is not None and Rvec.shape == (3, 1):
            R = cv2.Rodrigues(Rvec)[0]
        else:
            R = extri_fs.getNode(f'Rot_{name}').mat()
            Rvec = cv2.Rodrigues(R)[0]

        cam['R'] = R
        cam['T'] = Tvec.reshape(3, 1)
        cam['Rvec'] = Rvec

        # c2w from w2c
        w2c = np.eye(4, dtype=np.float32)
        w2c[:3, :3] = R
        w2c[:3, 3:] = Tvec.reshape(3, 1)
        c2w = np.linalg.inv(w2c)[:3]  # 3x4
        cam['c2w'] = c2w

        # Bounds
        n_node = extri_fs.getNode(f'n_{name}')
        f_node = extri_fs.getNode(f'f_{name}')
        cam['n'] = n_node.real() if n_node is not None and n_node.real() > 0 else 0.0001
        cam['f'] = f_node.real() if f_node is not None and f_node.real() > 0 else 1e6

        # Distortion
        D_node = intri_fs.getNode(f'D_{name}')
        cam['D'] = D_node.mat() if D_node is not None and D_node.mat() is not None else np.zeros((5, 1))

        cameras[name] = cam

    intri_fs.release()
    extri_fs.release()
    return names, cameras


def save_cameras_easyvolcap(output_dir, c2ws, K, H, W, n=0.0001, f=1e6, D=None):
    """Save interpolated cameras in EasyVolcap format (intri.yml + extri.yml)."""
    os.makedirs(output_dir, exist_ok=True)
    n_frames = len(c2ws)

    intri_path = join(output_dir, 'intri.yml')
    extri_path = join(output_dir, 'extri.yml')

    intri_fs = cv2.FileStorage(intri_path, cv2.FILE_STORAGE_WRITE)
    extri_fs = cv2.FileStorage(extri_path, cv2.FILE_STORAGE_WRITE)

    names = [f'{i:06d}' for i in range(n_frames)]
    intri_fs.write('names', names)
    extri_fs.write('names', names)

    if D is None:
        D = np.zeros((5, 1), dtype=np.float64)

    for i, name in enumerate(names):
        c2w = c2ws[i]  # 3x4
        # c2w -> w2c
        w2c = np.eye(4, dtype=np.float64)
        w2c[:3, :3] = c2w[:3, :3]
        w2c[:3, 3] = c2w[:3, 3]
        w2c = np.linalg.inv(w2c)

        R = w2c[:3, :3]
        T = w2c[:3, 3:]
        Rvec = cv2.Rodrigues(R)[0]

        intri_fs.write(f'K_{name}', K.astype(np.float64))
        intri_fs.write(f'H_{name}', float(H))
        intri_fs.write(f'W_{name}', float(W))
        intri_fs.write(f'D_{name}', D.astype(np.float64))

        extri_fs.write(f'R_{name}', Rvec.astype(np.float64))
        extri_fs.write(f'Rot_{name}', R.astype(np.float64))
        extri_fs.write(f'T_{name}', T.astype(np.float64))
        extri_fs.write(f'n_{name}', float(n))
        extri_fs.write(f'f_{name}', float(f))

    intri_fs.release()
    extri_fs.release()
    print(f'Saved {n_frames} cameras to {output_dir}')


# ────────────────────────── main ──────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Generate smooth camera path between two cameras')
    parser.add_argument('--data_root', type=str, required=True, help='EasyVolcap dataset root (containing intri.yml, extri.yml)')
    parser.add_argument('--cam_idx1', type=int, required=True, help='First camera index')
    parser.add_argument('--cam_idx2', type=int, required=True, help='Second camera index')
    parser.add_argument('--n_frames', type=int, default=120, help='Number of interpolated frames')
    parser.add_argument('--path_type', type=str, default='arc', choices=['arc', 'slerp', 'spiral'], help='Camera path type')
    parser.add_argument('--output', type=str, default=None, help='Output directory (default: data/camera_paths/<scene>_<idx1>_<idx2>)')
    args = parser.parse_args()

    # Load cameras
    names, cameras = load_cameras_from_easyvolcap(args.data_root)
    print(f'Loaded {len(names)} cameras from {args.data_root}')

    assert args.cam_idx1 < len(names), f'cam_idx1={args.cam_idx1} out of range [0, {len(names)})'
    assert args.cam_idx2 < len(names), f'cam_idx2={args.cam_idx2} out of range [0, {len(names)})'

    cam1 = cameras[names[args.cam_idx1]]
    cam2 = cameras[names[args.cam_idx2]]
    c2w1, c2w2 = cam1['c2w'], cam2['c2w']

    print(f'Camera {args.cam_idx1} -> Camera {args.cam_idx2}, path_type={args.path_type}, n_frames={args.n_frames}')

    # Generate path
    if args.path_type == 'arc':
        c2ws = generate_arc_path(c2w1, c2w2, args.n_frames)
    elif args.path_type == 'slerp':
        c2ws = generate_slerp_path(c2w1, c2w2, args.n_frames)
    elif args.path_type == 'spiral':
        c2ws = generate_spiral_segment_path(c2w1, c2w2, args.n_frames)
    else:
        raise ValueError(f'Unknown path_type: {args.path_type}')

    # Use intrinsics from camera 1 (original resolution — ratio is applied by config)
    K = cam1['K']
    H, W = cam1['H'], cam1['W']
    n_val, f_val = cam1['n'], cam1['f']
    D = cam1.get('D', None)

    # Output directory
    if args.output is None:
        scene_name = os.path.basename(os.path.normpath(args.data_root))
        args.output = f'data/camera_paths/{scene_name}_{args.cam_idx1}_{args.cam_idx2}_{args.path_type}'

    save_cameras_easyvolcap(args.output, c2ws, K, H, W, n=n_val, f=f_val, D=D)
    print(f'Done! Use with evc-test:')
    print(f'  evc-test -c <exp_config>,configs/specs/interp2cam.yaml \\')
    print(f'    val_dataloader_cfg.dataset_cfg.camera_path_intri={args.output}/intri.yml \\')
    print(f'    val_dataloader_cfg.dataset_cfg.camera_path_extri={args.output}/extri.yml \\')
    print(f'    val_dataloader_cfg.dataset_cfg.n_render_views={args.n_frames}')


if __name__ == '__main__':
    main()
