# SPDX-FileCopyrightText: Copyright (c) 2026 GLINT Authors
# SPDX-License-Identifier: MIT

"""Standalone loader for the EasyVolCap datasets used by GLINT.

This module deliberately does not import EasyVolCap.  It preserves the camera,
view split, image normalization, DiffusionRenderer prior, and temporal/view
neighbor semantics used by the released GLINT training configuration.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence, Union

import cv2
import numpy as np
import torch
import yaml
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from glint.camera import (
    GlintCamera,
    list_easyvolcap_cameras,
    load_easyvolcap_cameras,
)


PathLike = Union[str, Path]
_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".exr")
DEFAULT_DIFFREN_PARAMS = ("normal", "depth", "diffuse_albedo", "basecolor")


def decode_diffren_depth(depth_rgb: Tensor) -> Tensor:
    """Decode GLINT's normalized RGB depth into a scalar ``[..., 1]`` map."""

    if depth_rgb.shape[-1] != 3:
        if depth_rgb.shape[-1] == 1:
            return depth_rgb
        raise ValueError(f"Expected RGB or scalar depth, got {tuple(depth_rgb.shape)}")
    weights = depth_rgb.new_tensor((1.0, 1.0 / 256.0, 1.0 / (256.0**2)))
    return (depth_rgb * weights).sum(dim=-1, keepdim=True).clamp_(0.0, 1.0)


def _move_optional(value: Tensor | None, device: torch.device | str) -> Tensor | None:
    return None if value is None else value.to(device)


@dataclass
class GlintFrame:
    """One camera/frame and the supervision maps consumed by GLINT."""

    index: int
    camera_name: str
    frame_name: str
    camera: GlintCamera
    rgb: Tensor
    mask: Tensor | None
    stable_normal: Tensor | None
    diffren: dict[str, Tensor]
    sky_mask: Tensor | None
    paths: dict[str, Path]

    @property
    def normal(self) -> Tensor | None:
        return self.diffren.get("normal")

    @property
    def depth_rgb(self) -> Tensor | None:
        return self.diffren.get("depth")

    @property
    def depth(self) -> Tensor | None:
        return None if self.depth_rgb is None else decode_diffren_depth(self.depth_rgb)

    @property
    def diffuse_albedo(self) -> Tensor | None:
        return self.diffren.get("diffuse_albedo")

    @property
    def basecolor(self) -> Tensor | None:
        return self.diffren.get("basecolor")

    def to(self, device: torch.device | str) -> "GlintFrame":
        return replace(
            self,
            camera=self.camera.to(device),
            rgb=self.rgb.to(device),
            mask=_move_optional(self.mask, device),
            stable_normal=_move_optional(self.stable_normal, device),
            diffren={key: value.to(device) for key, value in self.diffren.items()},
            sky_mask=_move_optional(self.sky_mask, device),
        )


@dataclass
class GlintSample(GlintFrame):
    """A target frame plus GLINT's optional ``index +/- window`` neighbors."""

    previous: GlintFrame | None = None
    next: GlintFrame | None = None

    def to(self, device: torch.device | str) -> "GlintSample":
        return GlintSample(
            index=self.index,
            camera_name=self.camera_name,
            frame_name=self.frame_name,
            camera=self.camera.to(device),
            rgb=self.rgb.to(device),
            mask=_move_optional(self.mask, device),
            stable_normal=_move_optional(self.stable_normal, device),
            diffren={key: value.to(device) for key, value in self.diffren.items()},
            sky_mask=_move_optional(self.sky_mask, device),
            paths=self.paths,
            previous=None if self.previous is None else self.previous.to(device),
            next=None if self.next is None else self.next.to(device),
        )


@dataclass
class GlintBatch:
    """Collated samples with convenient stacked supervision tensors."""

    samples: tuple[GlintSample, ...]

    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self) -> Iterator[GlintSample]:
        return iter(self.samples)

    @property
    def cameras(self) -> tuple[GlintCamera, ...]:
        return tuple(sample.camera for sample in self.samples)

    @property
    def camera_names(self) -> tuple[str, ...]:
        return tuple(sample.camera_name for sample in self.samples)

    @property
    def rgb(self) -> Tensor:
        return torch.stack([sample.rgb for sample in self.samples])

    @property
    def stable_normal(self) -> Tensor | None:
        values = [sample.stable_normal for sample in self.samples]
        return None if any(value is None for value in values) else torch.stack(values)  # type: ignore[arg-type]

    @property
    def diffren(self) -> dict[str, Tensor]:
        if not self.samples:
            return {}
        keys = self.samples[0].diffren.keys()
        return {
            key: torch.stack([sample.diffren[key] for sample in self.samples])
            for key in keys
        }

    @property
    def depth(self) -> Tensor | None:
        depth = [sample.depth for sample in self.samples]
        return None if any(value is None for value in depth) else torch.stack(depth)  # type: ignore[arg-type]

    def to(self, device: torch.device | str) -> "GlintBatch":
        return GlintBatch(tuple(sample.to(device) for sample in self.samples))


def collate_glint_samples(samples: Sequence[GlintSample]) -> GlintBatch:
    if not samples:
        raise ValueError("Cannot collate an empty GLINT batch")
    shapes = {tuple(sample.rgb.shape) for sample in samples}
    if len(shapes) != 1:
        raise ValueError(f"A GLINT batch requires equal image shapes, got {shapes}")
    prior_keys = {tuple(sample.diffren.keys()) for sample in samples}
    if len(prior_keys) != 1:
        raise ValueError("A GLINT batch requires the same prior keys for every sample")
    return GlintBatch(tuple(samples))


def _deep_merge(base: Mapping[str, Any], update: Mapping[str, Any]) -> dict[str, Any]:
    output = dict(base)
    for key, value in update.items():
        if key == "_delete_":
            continue
        if isinstance(value, Mapping) and isinstance(output.get(key), Mapping):
            output[key] = _deep_merge(output[key], value)
        else:
            output[key] = value
    return output


def _read_yaml(path: Path) -> dict[str, Any]:
    text = path.read_text()
    # EasyVolCap permits VSCode-style placeholders that are not plain YAML.
    text = re.sub(
        r"(:\s*)(\{\{[^\n]+?\}\})(\s*(?:#.*)?)$", r'\1"\2"\3', text, flags=re.M
    )
    value = yaml.safe_load(text) or {}
    if not isinstance(value, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return value


def _resolve_config_include(config_path: Path, include: str) -> Path:
    include_path = Path(include)
    if include_path.is_absolute() and include_path.exists():
        return include_path
    # EasyVolCap include paths are rooted at an ancestor of the including file.
    # Prefer that local config tree over an unrelated ``configs/`` in the
    # process working directory (which is especially important for tools and
    # tests that load configs outside this repository).
    candidates = [parent / include_path for parent in config_path.parents]
    candidates.append(Path.cwd() / include_path)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(
        f"Could not resolve config include {include!r} from {config_path}"
    )


def load_easyvolcap_config(
    path: PathLike, _seen: set[Path] | None = None
) -> dict[str, Any]:
    """Resolve EasyVolCap's ``configs:`` inheritance into a plain mapping."""

    path = Path(path).resolve()
    seen = set() if _seen is None else _seen
    if path in seen:
        raise ValueError(f"Recursive EasyVolCap config include: {path}")
    seen.add(path)
    current = _read_yaml(path)
    includes = current.pop("configs", [])
    if isinstance(includes, str):
        includes = [includes]
    merged: dict[str, Any] = {}
    for include in includes:
        include_path = _resolve_config_include(path, str(include))
        merged = _deep_merge(merged, load_easyvolcap_config(include_path, seen))
    seen.remove(path)
    return _deep_merge(merged, current)


def _select_names(names: Sequence[str], sample: Sequence[Any] | None) -> list[str]:
    if sample is None:
        return list(names)
    sample = list(sample)
    if len(sample) == 3:
        start, stop, step = sample
        return list(names[slice(start, stop, step)])
    selected: list[str] = []
    for index in sample:
        index = int(index)
        try:
            selected.append(names[index])
        except IndexError as error:
            raise IndexError(
                f"Sample index {index} exceeds {len(names)} entries"
            ) from error
    return selected


def _resolve_image_path(directory: Path, frame_name: str) -> Path:
    direct = directory / frame_name
    if direct.suffix and direct.exists():
        return direct
    stem = direct.stem if direct.suffix else frame_name
    for extension in _IMAGE_EXTENSIONS:
        candidate = directory / f"{stem}{extension}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No image for frame {frame_name!r} under {directory}")


def _resolve_optional_image_path(directory: Path, frame_name: str) -> Path | None:
    try:
        return _resolve_image_path(directory, frame_name)
    except FileNotFoundError:
        return None


def _available_frames(directory: Path) -> list[str]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {directory}")
    frames = {
        path.stem
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in _IMAGE_EXTENSIONS
    }
    if not frames:
        raise FileNotFoundError(f"No supported images under {directory}")
    return sorted(
        frames,
        key=lambda value: (
            not value.isdigit(),
            int(value) if value.isdigit() else value,
        ),
    )


def _normalize_image(image: np.ndarray) -> Tensor:
    if image.ndim == 2:
        image = image[..., None]
    if image.shape[-1] >= 3:
        image = image[..., :3][..., ::-1].copy()  # OpenCV BGR -> RGB
    if np.issubdtype(image.dtype, np.integer):
        image = image.astype(np.float32) / float(np.iinfo(image.dtype).max)
    else:
        image = image.astype(np.float32)
    return torch.from_numpy(np.ascontiguousarray(image))


class GlintDataset(Dataset[GlintSample]):
    """Lazy, EasyVolCap-free dataset for GLINT scenes."""

    def __init__(
        self,
        data_root: PathLike,
        *,
        ratio: float = 1.0,
        split: str = "all",
        view_sample: Sequence[Any] | None = None,
        frame_sample: Sequence[Any] | None = None,
        frame_names: Sequence[str] | None = None,
        images_dir: str = "images",
        masks_dir: str = "masks",
        normals_dir: str = "normals",
        diffren_dir: str = "diffrens",
        sky_masks_dir: str = "sky_masks",
        load_mask: bool = False,
        load_stable_normal: bool = True,
        diffren_params: Sequence[str] = DEFAULT_DIFFREN_PARAMS,
        load_sky_mask: bool = False,
        load_neighbors: bool = False,
        neighbor_window: int = 5,
    ) -> None:
        if ratio <= 0:
            raise ValueError(f"ratio must be positive, got {ratio}")
        if neighbor_window < 0:
            raise ValueError("neighbor_window must be non-negative")
        self.data_root = Path(data_root).resolve()
        self.ratio = float(ratio)
        self.split = split.lower()
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.normals_dir = normals_dir
        self.diffren_dir = diffren_dir
        self.sky_masks_dir = sky_masks_dir
        self.load_mask = load_mask
        self.load_stable_normal = bool(
            load_stable_normal and (self.data_root / self.normals_dir).is_dir()
        )
        self.diffren_params = tuple(diffren_params)
        self.load_sky_mask = load_sky_mask
        self.load_neighbors = load_neighbors and self.split in {"train", "training"}
        self.neighbor_window = neighbor_window

        all_camera_names = list_easyvolcap_cameras(self.data_root)
        self.camera_names = _select_names(all_camera_names, view_sample)
        if not self.camera_names:
            raise ValueError("view_sample selected no cameras")
        self.cameras = load_easyvolcap_cameras(
            self.data_root,
            self.camera_names,
            ratio=self.ratio,
            device="cpu",
        )

        available_frames = _available_frames(
            self.data_root / self.images_dir / self.camera_names[0]
        )
        self.frame_names = (
            list(frame_names)
            if frame_names is not None
            else _select_names(available_frames, frame_sample)
        )
        if not self.frame_names:
            raise ValueError("frame_sample selected no frames")
        self.entries = [
            (camera_name, frame_name)
            for camera_name in self.camera_names
            for frame_name in self.frame_names
        ]

        self._undistortion: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        self._prepare_undistortion()
        # Fail early with a useful path error without eagerly decoding the dataset.
        self._paths(*self.entries[0])

    @classmethod
    def from_config(
        cls,
        config_path: PathLike,
        *,
        split: str = "train",
        **overrides: Any,
    ) -> "GlintDataset":
        """Build from either a dataset config or a full inherited experiment config."""

        config = load_easyvolcap_config(config_path)
        normalized_split = split.lower()
        loader_key = (
            "dataloader_cfg"
            if normalized_split in {"train", "training", "all"}
            else "val_dataloader_cfg"
        )
        try:
            dataset_config = config[loader_key]["dataset_cfg"]
        except KeyError as error:
            raise KeyError(
                f"{loader_key}.dataset_cfg is missing from {config_path}"
            ) from error

        use_diffren = bool(dataset_config.get("use_diffren", True))
        normal_source = (
            config.get("model_cfg", {})
            .get("supervisor_cfg", {})
            .get("use_normal_type", "diffren")
        )
        kwargs: dict[str, Any] = {
            "data_root": dataset_config["data_root"],
            "ratio": dataset_config.get("ratio", 1.0),
            "split": split,
            "view_sample": None
            if normalized_split == "all"
            else dataset_config.get("view_sample"),
            "frame_sample": dataset_config.get("frame_sample"),
            "images_dir": dataset_config.get("images_dir", "images"),
            "masks_dir": dataset_config.get("masks_dir", "masks"),
            "normals_dir": dataset_config.get("normals_dir", "normals"),
            "diffren_dir": dataset_config.get("diffren_dir", "diffrens"),
            "sky_masks_dir": dataset_config.get("sky_mask_dir", "sky_masks"),
            "load_mask": dataset_config.get("use_masks", False),
            "load_stable_normal": bool(dataset_config.get("use_normals", False))
            and normal_source == "stable",
            "diffren_params": dataset_config.get(
                "diffren_params", DEFAULT_DIFFREN_PARAMS
            )
            if use_diffren
            else (),
            "load_sky_mask": dataset_config.get("use_sky_masks", False),
            "load_neighbors": dataset_config.get("use_neighbors", False),
            "neighbor_window": dataset_config.get("neighbor_window", 5),
        }
        kwargs.update(overrides)
        return cls(**kwargs)

    def _prepare_undistortion(self) -> None:
        for name, camera in self.cameras.items():
            if camera.distortion is None or not torch.any(camera.distortion != 0):
                continue
            source_width = int(round(camera.image_width / self.ratio))
            source_height = int(round(camera.image_height / self.ratio))
            source_K = camera.K.cpu().numpy().copy()
            source_K[:2] /= self.ratio
            distortion = camera.distortion.cpu().numpy()
            new_K, _ = cv2.getOptimalNewCameraMatrix(
                source_K,
                distortion,
                (source_width, source_height),
                0,
                (source_width, source_height),
            )
            target_K = new_K.astype(np.float32)
            target_K[0] *= camera.image_width / source_width
            target_K[1] *= camera.image_height / source_height
            camera.K = torch.from_numpy(target_K)
            self._undistortion[name] = (source_K, distortion, new_K)

    def _paths(self, camera_name: str, frame_name: str) -> dict[str, Path]:
        paths = {
            "rgb": _resolve_image_path(
                self.data_root / self.images_dir / camera_name, frame_name
            )
        }
        if self.load_mask:
            paths["mask"] = _resolve_image_path(
                self.data_root / self.masks_dir / camera_name, frame_name
            )
        if self.load_stable_normal:
            stable_normal = _resolve_optional_image_path(
                self.data_root / self.normals_dir / camera_name, frame_name
            )
            if stable_normal is not None:
                paths["stable_normal"] = stable_normal
        for parameter in self.diffren_params:
            paths[f"diffren/{parameter}"] = _resolve_image_path(
                self.data_root / self.diffren_dir / parameter / camera_name,
                frame_name,
            )
        if self.load_sky_mask:
            paths["sky_mask"] = _resolve_image_path(
                self.data_root / self.sky_masks_dir / camera_name, frame_name
            )
        return paths

    def _load(self, path: Path, camera_name: str, *, grayscale: bool = False) -> Tensor:
        flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_UNCHANGED
        image = cv2.imread(str(path), flag)
        if image is None:
            raise OSError(f"OpenCV could not decode {path}")
        if camera_name in self._undistortion:
            source_K, distortion, new_K = self._undistortion[camera_name]
            image = cv2.undistort(image, source_K, distortion, newCameraMatrix=new_K)
        camera = self.cameras[camera_name]
        target_size = (camera.image_width, camera.image_height)
        if (image.shape[1], image.shape[0]) != target_size:
            interpolation = (
                cv2.INTER_AREA
                if target_size[0] <= image.shape[1] and target_size[1] <= image.shape[0]
                else cv2.INTER_LINEAR
            )
            image = cv2.resize(image, target_size, interpolation=interpolation)
        return _normalize_image(image)

    def _load_frame(self, index: int, *, include_neighbors: bool) -> GlintSample:
        camera_name, frame_name = self.entries[index]
        paths = self._paths(camera_name, frame_name)
        diffren = {
            parameter: self._load(paths[f"diffren/{parameter}"], camera_name)
            for parameter in self.diffren_params
        }
        frame = GlintSample(
            index=index,
            camera_name=camera_name,
            frame_name=frame_name,
            camera=self.cameras[camera_name],
            rgb=self._load(paths["rgb"], camera_name),
            mask=self._load(paths["mask"], camera_name, grayscale=True)
            if "mask" in paths
            else None,
            stable_normal=self._load(paths["stable_normal"], camera_name)
            if "stable_normal" in paths
            else None,
            diffren=diffren,
            sky_mask=self._load(paths["sky_mask"], camera_name, grayscale=True)
            if "sky_mask" in paths
            else None,
            paths=paths,
        )
        if include_neighbors and self.neighbor_window > 0:
            previous_index = max(0, index - self.neighbor_window)
            next_index = min(len(self) - 1, index + self.neighbor_window)
            frame.previous = self._load_frame(previous_index, include_neighbors=False)
            frame.next = self._load_frame(next_index, include_neighbors=False)
        return frame

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> GlintSample:
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        return self._load_frame(index, include_neighbors=self.load_neighbors)

    def summary(self) -> dict[str, Any]:
        camera = self.cameras[self.camera_names[0]]
        return {
            "data_root": str(self.data_root),
            "split": self.split,
            "views": len(self.camera_names),
            "frames_per_view": len(self.frame_names),
            "samples": len(self),
            "resolution": [camera.image_height, camera.image_width],
            "stable_normal": self.load_stable_normal,
            "diffren_params": list(self.diffren_params),
            "neighbors": self.load_neighbors,
            "neighbor_window": self.neighbor_window,
        }


def make_glint_dataloader(
    dataset: Dataset[GlintSample],
    *,
    batch_size: int = 1,
    shuffle: bool | None = None,
    num_workers: int = 0,
    pin_memory: bool = False,
    **kwargs: Any,
) -> DataLoader[GlintBatch]:
    """Construct a DataLoader using GLINT's structured camera-aware collate."""

    if shuffle is None:
        shuffle = getattr(dataset, "split", "") in {"train", "training"}
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_glint_samples,
        **kwargs,
    )
