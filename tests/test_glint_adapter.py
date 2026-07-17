# SPDX-FileCopyrightText: Copyright (c) 2026 GLINT Authors
# SPDX-License-Identifier: MIT

from types import SimpleNamespace

import pytest
import torch

from glint.checkpoint import load_glint_checkpoint
from glint.renderer import rasterize_glint_2dgs, render_glint_camera


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="GLINT adapter requires CUDA"
)


def _scene(n: int = 24, height: int = 48, width: int = 64):
    torch.manual_seed(23)
    device = torch.device("cuda")
    means = torch.cat(
        (
            torch.empty(n, 2, device=device).uniform_(-0.5, 0.5),
            torch.empty(n, 1, device=device).uniform_(2.0, 3.0),
        ),
        dim=-1,
    ).requires_grad_()
    quats = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).repeat(n, 1)
    quats.requires_grad_()
    scales = torch.full((n, 2), 0.16, device=device, requires_grad=True)
    opacities = torch.full((n, 1), 0.7, device=device, requires_grad=True)
    features = torch.rand(n, 5, device=device, requires_grad=True)
    viewmat = torch.eye(4, device=device)
    K = torch.tensor(
        [[55.0, 0.0, width / 2], [0.0, 55.0, height / 2], [0.0, 0.0, 1.0]],
        device=device,
    )
    return SimpleNamespace(
        n=n,
        height=height,
        width=width,
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        features=features,
        viewmat=viewmat,
        K=K,
    )


def test_glint_material_render_and_legacy_gradient():
    scene = _scene()
    output = rasterize_glint_2dgs(
        means=scene.means,
        quats=scene.quats,
        scales=scene.scales,
        opacities=scene.opacities,
        features=scene.features,
        viewmat=scene.viewmat,
        K=scene.K,
        width=scene.width,
        height=scene.height,
        background=torch.zeros(3, device="cuda"),
        specular_channels=1,
    )

    assert output.render.shape == (3, scene.height, scene.width)
    assert output.specular.shape == (1, scene.height, scene.width)
    assert output.transparency.shape == (1, scene.height, scene.width)
    assert output.conditional_transparency.shape == (1, scene.height, scene.width)
    assert output.rend_alpha.shape == (1, scene.height, scene.width)
    assert output.rend_normal.shape == (3, scene.height, scene.width)
    assert output.surf_depth.shape == (1, scene.height, scene.width)
    assert output.surf_normal.shape == (3, scene.height, scene.width)
    assert output.radii.shape == (scene.n,)
    assert output.visibility_filter.shape == (scene.n,)
    assert "weight_accumulate" not in output

    loss = (
        output.render.square().mean()
        + output.specular.square().mean()
        + output.transparency.square().mean()
        + output.rend_alpha.mean()
        + 0.01 * output.rend_dist.mean()
    )
    loss.backward()
    for value in (
        scene.means,
        scene.quats,
        scene.scales,
        scene.opacities,
        scene.features,
    ):
        assert value.grad is not None
        assert torch.isfinite(value.grad).all()
    assert output.viewspace_points.grad is not None
    assert output.viewspace_points.grad.shape == (scene.n, 3)
    assert torch.isfinite(output.viewspace_points.grad).all()


def test_conditional_transparency_gradient_is_isolated_from_geometry():
    scene = _scene()
    output = rasterize_glint_2dgs(
        means=scene.means,
        quats=scene.quats,
        scales=scene.scales,
        opacities=scene.opacities,
        features=scene.features,
        viewmat=scene.viewmat,
        K=scene.K,
        width=scene.width,
        height=scene.height,
        background=torch.zeros(3, device="cuda"),
        specular_channels=1,
    )

    geometry_and_material = (
        scene.means,
        scene.quats,
        scene.scales,
        scene.opacities,
        scene.features,
    )
    gradients = torch.autograd.grad(
        output.conditional_transparency.sum(),
        geometry_and_material,
        allow_unused=True,
    )
    for gradient in gradients[:-1]:
        assert gradient is None or torch.allclose(
            gradient, torch.zeros_like(gradient), atol=2e-5
        )
    feature_gradient = gradients[-1]
    assert feature_gradient is not None
    assert torch.count_nonzero(feature_gradient[:, -1]) > 0
    assert torch.allclose(
        feature_gradient[:, :-1],
        torch.zeros_like(feature_gradient[:, :-1]),
        atol=2e-6,
    )

    coverage = output.rend_alpha.detach()
    valid = coverage > 1e-4
    assert torch.allclose(
        output.conditional_transparency[valid] * coverage[valid],
        output.transparency[valid],
        atol=2e-5,
    )


def test_glint_trans_mask_restores_full_gaussian_indexing():
    scene = _scene()
    mask = torch.zeros(scene.n, dtype=torch.bool, device="cuda")
    mask[::2] = True
    output = rasterize_glint_2dgs(
        means=scene.means,
        quats=scene.quats,
        scales=scene.scales,
        opacities=scene.opacities,
        features=scene.features,
        viewmat=scene.viewmat,
        K=scene.K,
        width=scene.width,
        height=scene.height,
        trans_mask=mask,
    )
    output.render.mean().backward()

    assert not output.visibility_filter[~mask].any()
    assert torch.count_nonzero(output.radii[~mask]) == 0
    assert torch.count_nonzero(scene.features.grad[~mask]) == 0
    assert torch.count_nonzero(output.viewspace_points.grad[~mask]) == 0


def test_duck_typed_glint_camera_and_model_entry_point():
    scene = _scene()
    # Degree-zero SH coefficients that remain differentiable through the adapter.
    sh_coeffs = torch.rand(scene.n, 1, 3, device="cuda", requires_grad=True)
    model = SimpleNamespace(
        get_xyz=scene.means,
        get_rotation=scene.quats,
        get_scaling=scene.scales,
        get_opacity=scene.opacities,
        get_features=sh_coeffs,
        get_specular=torch.full((scene.n, 1), 0.1, device="cuda", requires_grad=True),
        get_transmission_coeff=torch.full(
            (scene.n, 1), 0.3, device="cuda", requires_grad=True
        ),
        active_sh_degree=torch.tensor(0, device="cuda"),
        render_reflection=True,
        specular_channels=1,
    )
    camera = SimpleNamespace(
        world_view_transform=scene.viewmat.T,
        K=scene.K,
        camera_center=torch.zeros(3, device="cuda"),
        image_width=scene.width,
        image_height=scene.height,
        znear=0.01,
        zfar=100.0,
    )
    pipe = SimpleNamespace(compute_cov3D_python=False, depth_ratio=0.0)
    output = render_glint_camera(camera, model, pipe, torch.zeros(3, device="cuda"))
    output.render.mean().backward()

    assert output.render.shape == (3, scene.height, scene.width)
    assert sh_coeffs.grad is not None
    assert torch.isfinite(sh_coeffs.grad).all()


def test_glint_checkpoint_loader_preserves_parameterization(tmp_path):
    n = 3
    state = {
        "sampler.bg_color": torch.zeros(3),
        "sampler.env_bg_color": torch.zeros(3),
        "sampler.trans_env_bg_color": torch.zeros(3),
        "sampler.pcd.active_sh_degree": torch.tensor([1]),
        "sampler.pcd._xyz": torch.randn(n, 3),
        "sampler.pcd._features_dc": torch.randn(n, 1, 3),
        "sampler.pcd._features_rest": torch.randn(n, 3, 3),
        "sampler.pcd._scaling": torch.randn(n, 2),
        "sampler.pcd._rotation": torch.randn(n, 4),
        "sampler.pcd._opacity": torch.randn(n, 1),
        "sampler.pcd._specular": torch.randn(n, 1),
        "sampler.pcd._roughness": torch.randn(n, 1),
        "sampler.pcd._transmission_coeff": torch.randn(n, 1),
        "sampler.pcd._ior": torch.randn(n, 1),
    }
    path = tmp_path / "checkpoint.pt"
    torch.save({"model": state, "epoch": 17}, path)
    checkpoint = load_glint_checkpoint(path, gaussian_sets=("pcd",))

    assert checkpoint.epoch == 17
    assert checkpoint.pcd.get_xyz.shape == (n, 3)
    assert checkpoint.pcd.get_features.shape == (n, 4, 3)
    assert torch.allclose(
        checkpoint.pcd.get_scaling, state["sampler.pcd._scaling"].exp()
    )
    assert torch.allclose(
        checkpoint.pcd.get_opacity, state["sampler.pcd._opacity"].sigmoid()
    )
    assert not hasattr(checkpoint.pcd, "_roughness")
    assert not hasattr(checkpoint.pcd, "_ior")
