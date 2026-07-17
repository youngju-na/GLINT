# SPDX-FileCopyrightText: Copyright (c) 2026 GLINT Authors
# SPDX-License-Identifier: MIT

import shutil
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from glint.camera import (
    list_easyvolcap_cameras,
    load_easyvolcap_camera,
    load_easyvolcap_cameras,
)
from glint.dataset import (
    GlintDataset,
    decode_diffren_depth,
    load_easyvolcap_config,
    make_glint_dataloader,
)


def _write_names(storage: cv2.FileStorage, names: list[str]) -> None:
    storage.startWriteStruct("names", cv2.FileNode_SEQ)
    for name in names:
        storage.write("", name)
    storage.endWriteStruct()


def _make_dataset(root: Path, views: int = 4) -> list[str]:
    root.mkdir(parents=True, exist_ok=True)
    names = [f"{index:04d}" for index in range(views)]
    intrinsics = cv2.FileStorage(str(root / "intri.yml"), cv2.FILE_STORAGE_WRITE)
    extrinsics = cv2.FileStorage(str(root / "extri.yml"), cv2.FILE_STORAGE_WRITE)
    _write_names(intrinsics, names)
    _write_names(extrinsics, names)
    for index, name in enumerate(names):
        intrinsics.write(
            f"K_{name}",
            np.array([[8.0, 0.0, 4.0], [0.0, 8.0, 3.0], [0.0, 0.0, 1.0]]),
        )
        intrinsics.write(f"D_{name}", np.zeros((5, 1)))
        intrinsics.write(f"H_{name}", 6.0)
        intrinsics.write(f"W_{name}", 8.0)
        extrinsics.write(f"R_{name}", np.zeros((3, 1)))
        extrinsics.write(f"T_{name}", np.array([[float(index)], [0.0], [0.0]]))
        extrinsics.write(f"n_{name}", 0.01)
        extrinsics.write(f"f_{name}", 100.0)
    intrinsics.release()
    extrinsics.release()

    for index, name in enumerate(names):
        image = np.zeros((6, 8, 3), dtype=np.uint8)
        image[..., 0] = 10 + index  # B
        image[..., 1] = 20 + index  # G
        image[..., 2] = 30 + index  # R
        normal = np.full((6, 8, 3), 127 + index, dtype=np.uint8)
        depth = np.empty((6, 8, 3), dtype=np.uint8)
        depth[...] = (64, 128, 255)  # BGR -> RGB = (255, 128, 64)
        for relative, value in (
            ("images", image),
            ("normals", normal),
            ("diffrens/normal", normal),
            ("diffrens/depth", depth),
            ("diffrens/diffuse_albedo", image),
            ("diffrens/basecolor", image),
        ):
            directory = root / relative / name
            directory.mkdir(parents=True, exist_ok=True)
            assert cv2.imwrite(str(directory / "000000.png"), value)
    return names


def test_bulk_easyvolcap_camera_loader(tmp_path: Path):
    names = _make_dataset(tmp_path)
    assert list_easyvolcap_cameras(tmp_path) == names

    cameras = load_easyvolcap_cameras(tmp_path, ("0001", "0003"), ratio=0.5)
    assert list(cameras) == ["0001", "0003"]
    assert cameras["0001"].image_height == 3
    assert cameras["0001"].image_width == 4
    assert torch.allclose(
        cameras["0001"].K,
        torch.tensor([[4.0, 0.0, 2.0], [0.0, 4.0, 1.5], [0.0, 0.0, 1.0]]),
    )
    assert torch.allclose(cameras["0003"].camera_center, torch.tensor([-3.0, 0.0, 0.0]))
    assert load_easyvolcap_camera(tmp_path, "0003", ratio=0.5).name == "0003"


def test_glint_dataset_priors_neighbors_and_dataloader(tmp_path: Path):
    _make_dataset(tmp_path)
    dataset = GlintDataset(
        tmp_path,
        ratio=0.5,
        split="train",
        view_sample=[1, 3],
        load_neighbors=True,
        neighbor_window=1,
    )
    assert dataset.camera_names == ["0001", "0003"]
    assert dataset.summary()["resolution"] == [3, 4]
    sample = dataset[0]
    assert sample.rgb.shape == (3, 4, 3)
    assert sample.rgb[0, 0].tolist() == pytest.approx([31 / 255, 21 / 255, 11 / 255])
    assert sample.stable_normal is not None
    assert set(sample.diffren) == {"normal", "depth", "diffuse_albedo", "basecolor"}
    expected_depth = 1.0 + (128 / 255) / 256 + (64 / 255) / (256**2)
    assert sample.depth is not None
    assert sample.depth[0, 0, 0].item() == pytest.approx(1.0, abs=1e-7)
    assert expected_depth > 1.0  # the released decoder clamps this encoding to one
    assert sample.previous is not None and sample.previous.index == 0
    assert sample.next is not None and sample.next.index == 1

    batch = next(iter(make_glint_dataloader(dataset, batch_size=2, shuffle=False)))
    assert batch.camera_names == ("0001", "0003")
    assert batch.rgb.shape == (2, 3, 4, 3)
    assert batch.depth is not None and batch.depth.shape == (2, 3, 4, 1)
    moved = batch.to("cpu")
    assert moved.samples[0].previous is not None


def test_missing_stable_normals_are_optional(tmp_path: Path):
    _make_dataset(tmp_path)
    shutil.rmtree(tmp_path / "normals")

    dataset = GlintDataset(tmp_path, load_stable_normal=True)
    sample = dataset[0]

    assert not dataset.load_stable_normal
    assert sample.stable_normal is None
    assert sample.normal is not None


def test_depth_decoder_and_inherited_easyvolcap_config(tmp_path: Path):
    data_root = tmp_path / "scene"
    _make_dataset(data_root)
    configs = tmp_path / "configs"
    experiments = configs / "experiments"
    experiments.mkdir(parents=True)
    (configs / "base.yaml").write_text(
        """
dataloader_cfg:
  dataset_cfg: &dataset
    ratio: 1.0
    frame_sample: [0, null, 1]
    use_normals: true
    use_diffren: true
    diffren_params: [normal, depth, diffuse_albedo, basecolor]
val_dataloader_cfg:
  dataset_cfg:
    <<: *dataset
"""
    )
    (experiments / "scene.yaml").write_text(
        f"""
configs: configs/base.yaml
dataloader_cfg:
  dataset_cfg:
    data_root: {data_root}
    view_sample: [1, 2]
val_dataloader_cfg:
  dataset_cfg:
    data_root: {data_root}
    view_sample: [0, 3]
exp_name: {{{{fileBasenameNoExtension}}}}
"""
    )

    config = load_easyvolcap_config(experiments / "scene.yaml")
    assert config["dataloader_cfg"]["dataset_cfg"]["use_diffren"]
    train = GlintDataset.from_config(experiments / "scene.yaml", split="train")
    val = GlintDataset.from_config(experiments / "scene.yaml", split="val")
    assert train.camera_names == ["0001", "0002"]
    assert val.camera_names == ["0000", "0003"]
    assert not train.load_stable_normal

    encoded = torch.tensor([[[1.0, 0.5, 0.25]]])
    decoded = decode_diffren_depth(encoded)
    assert decoded.shape == (1, 1, 1)
    assert decoded.item() == 1.0
