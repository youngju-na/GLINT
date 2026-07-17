# SPDX-FileCopyrightText: Copyright (c) 2026 GLINT Authors
# SPDX-License-Identifier: MIT

"""Minimal reader for EasyVolCap's OpenCV camera files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Union

import cv2
import numpy as np
import torch
from torch import Tensor


@dataclass
class GlintCamera:
    """Single GLINT camera with the attributes used by the renderer adapter."""

    K: Tensor
    R: Tensor
    T: Tensor
    image_width: int
    image_height: int
    znear: float = 0.0001
    zfar: float = 1e6
    name: str = ""
    distortion: Tensor | None = None

    @property
    def world_view_transform(self) -> Tensor:
        viewmat = torch.eye(4, dtype=self.R.dtype, device=self.R.device)
        viewmat[:3, :3] = self.R
        viewmat[:3, 3:] = self.T.reshape(3, 1)
        return viewmat.T.contiguous()

    @property
    def camera_center(self) -> Tensor:
        return (-self.R.T @ self.T.reshape(3, 1)).reshape(3)

    def get_k(self, scale: float = 1.0) -> Tensor:
        if scale == 1.0:
            return self.K
        K = self.K.clone()
        K[:2] /= scale
        return K

    def to(self, device: Union[str, torch.device]) -> "GlintCamera":
        return GlintCamera(
            K=self.K.to(device),
            R=self.R.to(device),
            T=self.T.to(device),
            image_width=self.image_width,
            image_height=self.image_height,
            znear=self.znear,
            zfar=self.zfar,
            name=self.name,
            distortion=None if self.distortion is None else self.distortion.to(device),
        )


def _matrix(storage: cv2.FileStorage, key: str) -> np.ndarray | None:
    node = storage.getNode(key)
    if node.empty():
        return None
    return node.mat()


def _real(storage: cv2.FileStorage, key: str, default: float) -> float:
    node = storage.getNode(key)
    return default if node.empty() else float(node.real())


def _camera_names(storage: cv2.FileStorage) -> list[str]:
    node = storage.getNode("names")
    if node.empty():
        raise KeyError("names is missing from intri.yml")
    return [node.at(index).string() for index in range(node.size())]


def list_easyvolcap_cameras(data_root: Union[str, Path]) -> list[str]:
    """Return camera names in the order stored by EasyVolCap."""

    path = Path(data_root) / "intri.yml"
    storage = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    if not storage.isOpened():
        storage.release()
        raise FileNotFoundError(f"Could not open {path}")
    try:
        return _camera_names(storage)
    finally:
        storage.release()


def load_easyvolcap_cameras(
    data_root: Union[str, Path],
    camera_names: Iterable[str] | None = None,
    ratio: float = 1.0,
    device: Union[str, torch.device] = "cpu",
) -> dict[str, GlintCamera]:
    """Load multiple cameras while opening each OpenCV YAML file only once."""

    if ratio <= 0:
        raise ValueError(f"ratio must be positive, got {ratio}")
    data_root = Path(data_root)
    intrinsics = cv2.FileStorage(str(data_root / "intri.yml"), cv2.FILE_STORAGE_READ)
    extrinsics = cv2.FileStorage(str(data_root / "extri.yml"), cv2.FILE_STORAGE_READ)
    if not intrinsics.isOpened() or not extrinsics.isOpened():
        intrinsics.release()
        extrinsics.release()
        raise FileNotFoundError(f"Could not open EasyVolCap cameras under {data_root}")

    try:
        available = _camera_names(intrinsics)
        names = available if camera_names is None else list(camera_names)
        missing = sorted(set(names) - set(available))
        if missing:
            raise KeyError(f"Unknown EasyVolCap cameras: {missing}")

        cameras: dict[str, GlintCamera] = {}
        for camera_name in names:
            K = _matrix(intrinsics, f"K_{camera_name}")
            if K is None:
                raise KeyError(f"K_{camera_name} is missing from intri.yml")
            height = int(_real(intrinsics, f"H_{camera_name}", -1) * ratio)
            width = int(_real(intrinsics, f"W_{camera_name}", -1) * ratio)
            if height <= 0 or width <= 0:
                raise ValueError(f"Invalid image size for camera {camera_name}")

            rotation_vector = _matrix(extrinsics, f"R_{camera_name}")
            if rotation_vector is not None:
                R = cv2.Rodrigues(rotation_vector)[0]
            else:
                R = _matrix(extrinsics, f"Rot_{camera_name}")
            T = _matrix(extrinsics, f"T_{camera_name}")
            if R is None or T is None:
                raise KeyError(f"Extrinsics for camera {camera_name} are incomplete")
            distortion = _matrix(intrinsics, f"D_{camera_name}")
            if distortion is None:
                distortion = _matrix(intrinsics, f"dist_{camera_name}")
            if distortion is None:
                distortion = np.zeros((5, 1), dtype=np.float32)

            scaled_K = K.astype(np.float32)
            scaled_K[:2] *= ratio
            cameras[camera_name] = GlintCamera(
                K=torch.from_numpy(scaled_K).to(device),
                R=torch.from_numpy(R.astype(np.float32)).to(device),
                T=torch.from_numpy(T.astype(np.float32)).reshape(3, 1).to(device),
                image_width=width,
                image_height=height,
                znear=_real(extrinsics, f"n_{camera_name}", 0.0001),
                zfar=_real(extrinsics, f"f_{camera_name}", 1e6),
                name=camera_name,
                distortion=torch.from_numpy(distortion.astype(np.float32))
                .reshape(-1)
                .to(device),
            )
        return cameras
    finally:
        intrinsics.release()
        extrinsics.release()


def load_easyvolcap_camera(
    data_root: Union[str, Path],
    camera_name: str = "0000",
    ratio: float = 1.0,
    device: Union[str, torch.device] = "cpu",
) -> GlintCamera:
    """Load one camera from ``intri.yml`` and ``extri.yml`` without EasyVolCap."""

    return load_easyvolcap_cameras(
        data_root,
        camera_names=(camera_name,),
        ratio=ratio,
        device=device,
    )[camera_name]
