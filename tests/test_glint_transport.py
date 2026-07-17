# SPDX-FileCopyrightText: Copyright (c) 2026 GLINT Authors
# SPDX-License-Identifier: MIT

import importlib.util

import pytest
import torch

from glint.camera import GlintCamera
from glint.checkpoint import GlintCheckpoint, GlintGaussianSet
from glint.geometry import quaternion_to_rotation_matrix
from glint.optix_backend import make_surfel_triangles
from glint.torch_tracer import TorchSurfelTracer
from glint.transport import (
    RayBundle,
    compose_glint_radiance,
    compute_transport_weights,
    generate_camera_rays,
    make_surface_rays,
    render_glint_transport,
)


def test_quaternion_to_rotation_matrix_is_orthonormal() -> None:
    quaternions = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0], [0.5, -0.2, 0.3, 0.4]],
        dtype=torch.float64,
    )
    rotations = quaternion_to_rotation_matrix(quaternions)
    identity = torch.eye(3, dtype=rotations.dtype).expand_as(rotations)

    torch.testing.assert_close(rotations.transpose(-1, -2) @ rotations, identity)
    torch.testing.assert_close(
        torch.linalg.det(rotations), torch.ones(2, dtype=rotations.dtype)
    )


def _logit(value: float, *, device: torch.device) -> torch.Tensor:
    return torch.logit(torch.tensor(value, device=device))


def _gaussian_set(
    xyz,
    *,
    color=(0.5, 0.5, 0.5),
    scale=(1.0, 1.0),
    opacity=0.9,
    specularity=0.2,
    transparency=0.5,
    role="reflection",
    trainable=True,
    device=torch.device("cpu"),
) -> GlintGaussianSet:
    xyz = torch.as_tensor(xyz, dtype=torch.float32, device=device).reshape(-1, 3)
    n = xyz.shape[0]
    color = torch.as_tensor(color, dtype=torch.float32, device=device)
    dc = ((color - 0.5) / 0.28209479177387814).reshape(1, 1, 3).repeat(n, 1, 1)
    state = {
        "g.active_sh_degree": torch.tensor([0], device=device),
        "g._xyz": xyz,
        "g._features_dc": dc,
        "g._features_rest": torch.empty(n, 0, 3, device=device),
        "g._scaling": torch.tensor(scale, device=device)
        .log()
        .reshape(1, 2)
        .repeat(n, 1),
        "g._rotation": torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)
        .reshape(1, 4)
        .repeat(n, 1),
        "g._opacity": _logit(opacity, device=device).reshape(1, 1).repeat(n, 1),
        "g._specular": _logit(specularity, device=device).reshape(1, 1).repeat(n, 1),
        "g._transmission_coeff": _logit(transparency, device=device)
        .reshape(1, 1)
        .repeat(n, 1),
    }
    return GlintGaussianSet(
        state,
        "g",
        role=role,
        trainable=trainable,
    )


def test_camera_and_surface_ray_conventions():
    camera = GlintCamera(
        K=torch.eye(3),
        R=torch.eye(3),
        T=torch.zeros(3, 1),
        image_width=2,
        image_height=1,
    )
    rays = generate_camera_rays(camera)
    assert torch.allclose(
        rays.directions,
        torch.tensor([[[0.5, 0.5, 1.0], [1.5, 0.5, 1.0]]]),
    )

    depth = torch.full((1, 2, 1), 2.0, requires_grad=True)
    normal = torch.tensor([[[0.0, 0.0, -1.0], [0.0, 0.0, -1.0]]])
    transmitted, reflected = make_surface_rays(rays, depth, normal)
    assert torch.allclose(transmitted.origins, 2.0 * rays.directions)
    assert torch.allclose(transmitted.directions, rays.directions)
    assert torch.allclose(reflected.directions[..., 2], -torch.ones(1, 2))
    assert not transmitted.origins.requires_grad


def test_paper_transport_weights_are_energy_conserving_and_differentiable():
    directions = torch.tensor([[[0.0, 0.0, 1.0]]])
    normals = torch.tensor([[[0.0, 0.0, -1.0]]], requires_grad=True)
    transparency = torch.tensor([[[0.6]]], requires_grad=True)
    specularity = torch.tensor([[[0.2]]], requires_grad=True)
    weights = compute_transport_weights(
        view_directions=directions,
        normals=normals,
        transparency=transparency,
        specularity=specularity,
    )

    expected_reflection = 0.2 + 0.8 * 0.04
    assert torch.allclose(weights.reflection, torch.tensor([[[expected_reflection]]]))
    assert torch.allclose(
        weights.diffuse, torch.tensor([[[(1.0 - expected_reflection) * 0.4]]])
    )
    assert torch.allclose(
        weights.transmission, torch.tensor([[[(1.0 - expected_reflection) * 0.6]]])
    )
    assert torch.allclose(weights.energy_sum, torch.ones_like(weights.energy_sum))

    secondary_weights = compute_transport_weights(
        view_directions=directions,
        normals=normals,
        transparency=transparency,
        specularity=specularity,
        transmission_specularity=torch.tensor([[[0.25]]]),
    )
    initial_transmission = (1.0 - expected_reflection) * 0.6
    assert torch.allclose(
        secondary_weights.secondary_reflection,
        torch.tensor([[[initial_transmission * 0.25]]]),
    )
    assert torch.allclose(
        secondary_weights.transmission,
        torch.tensor([[[initial_transmission * 0.75]]]),
    )
    assert torch.allclose(
        secondary_weights.energy_sum,
        torch.ones_like(secondary_weights.energy_sum),
    )

    composition = compose_glint_radiance(
        interface_rgb=torch.ones(1, 1, 3, requires_grad=True),
        reflection_rgb=torch.full((1, 1, 3), 2.0, requires_grad=True),
        transmission_rgb=torch.full((1, 1, 3), 3.0, requires_grad=True),
        view_directions=directions,
        normals=normals,
        transparency=transparency,
        specularity=specularity,
    )
    composition.rgb.sum().backward()
    assert transparency.grad is not None
    assert specularity.grad is not None
    assert normals.grad is not None


def test_surfel_triangle_extent_matches_glint_bvh_geometry():
    vertices, triangles = make_surfel_triangles(
        torch.zeros(1, 3),
        torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        torch.tensor([[2.0, 1.0]]),
    )
    assert torch.allclose(vertices[0], torch.tensor([-6.0, 3.0, 0.0]))
    assert torch.equal(
        triangles, torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.int32)
    )


def test_torch_surfel_tracer_forward_backward():
    gaussian = _gaussian_set(
        [[0.0, 0.0, 2.0]],
        color=(0.8, 0.2, 0.1),
        role="interface",
    )
    rays = RayBundle(
        origins=torch.zeros(1, 1, 3),
        directions=torch.tensor([[[0.0, 0.0, 1.0]]]),
    )
    result = TorchSurfelTracer().trace(gaussian, rays, background=torch.zeros(3))
    assert result.rgb.shape == (1, 1, 3)
    assert torch.allclose(result.depth, torch.tensor([[[1.8]]]), atol=1e-5)
    assert torch.allclose(result.alpha, torch.tensor([[[0.9]]]), atol=1e-5)
    assert result.specular is not None
    assert result.transparency is not None

    loss = (
        result.rgb.sum()
        + result.depth.sum()
        + result.alpha.sum()
        + result.specular.sum()
    )
    loss.backward()
    for parameter in (
        gaussian._xyz,
        gaussian._features_dc,
        gaussian._opacity,
        gaussian._specular,
    ):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or importlib.util.find_spec("diff_surfel_tracing") is None,
    reason="optional GLINT OptiX extension is unavailable",
)
def test_optix_backend_matches_reference_single_surfel():
    from glint.optix_backend import OptixSurfelTracer

    device = torch.device("cuda")
    gaussian = _gaussian_set(
        [[0.0, 0.0, 2.0]],
        color=(0.8, 0.2, 0.1),
        role="interface",
        device=device,
    )
    rays = RayBundle(
        origins=torch.zeros(1, 1, 3, device=device),
        directions=torch.tensor([[[0.0, 0.0, 1.0]]], device=device),
    )
    result = OptixSurfelTracer().trace(
        gaussian, rays, background=torch.zeros(3, device=device)
    )
    assert torch.allclose(
        result.rgb, torch.tensor([[[0.72, 0.18, 0.09]]], device=device), atol=1e-5
    )
    assert torch.allclose(
        result.depth, torch.tensor([[[1.8]]], device=device), atol=1e-5
    )
    assert torch.allclose(
        result.alpha, torch.tensor([[[0.9]]], device=device), atol=1e-5
    )

    (result.rgb.sum() + result.depth.sum() + result.alpha.sum()).backward()
    assert gaussian._xyz.grad is not None
    assert gaussian._features_dc.grad is not None
    assert gaussian._opacity.grad is not None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="gsplat 2DGS requires CUDA")
def test_full_three_set_transport_forward_backward():
    device = torch.device("cuda")
    pcd = _gaussian_set(
        [[0.0, 0.0, 2.0]],
        color=(0.4, 0.3, 0.2),
        scale=(0.8, 0.8),
        opacity=0.95,
        specularity=0.15,
        transparency=0.7,
        role="interface",
        device=device,
    )
    reflection = _gaussian_set(
        [[0.0, 0.0, -2.0]],
        color=(0.1, 0.7, 0.2),
        scale=(5.0, 5.0),
        opacity=0.9,
        device=device,
    )
    transmission = _gaussian_set(
        [[0.0, 0.0, 4.0]],
        color=(0.2, 0.3, 0.9),
        scale=(5.0, 5.0),
        opacity=0.9,
        role="transmission",
        device=device,
    )
    checkpoint = GlintCheckpoint(
        pcd=pcd,
        env=reflection,
        trans_env=transmission,
        bg_color=torch.zeros(3, device=device),
        env_bg_color=torch.zeros(3, device=device),
        trans_env_bg_color=torch.zeros(3, device=device),
        epoch=0,
    )
    camera = GlintCamera(
        K=torch.tensor(
            [[8.0, 0.0, 4.0], [0.0, 8.0, 4.0], [0.0, 0.0, 1.0]],
            device=device,
        ),
        R=torch.eye(3, device=device),
        T=torch.zeros(3, 1, device=device),
        image_width=8,
        image_height=8,
    )
    interface_output = render_glint_transport(
        camera,
        checkpoint,
        None,
        stage="interface",
    )
    assert interface_output.stage == "interface"
    assert interface_output.transmission_trace is None
    assert interface_output.reflection_trace is None

    transmission_output = render_glint_transport(
        camera,
        checkpoint,
        TorchSurfelTracer(max_ray_gaussian_pairs=10_000),
        stage="transmission",
    )
    assert transmission_output.stage == "transmission"
    assert transmission_output.transmission_trace is not None
    assert transmission_output.reflection_trace is None
    assert torch.allclose(
        transmission_output.energy_sum,
        torch.ones_like(transmission_output.energy_sum),
        atol=1e-6,
    )
    transmission_weights = compute_transport_weights(
        view_directions=generate_camera_rays(camera).directions,
        normals=transmission_output.norm_map.reshape(8, 8, 3),
        transparency=transmission_output.trans_map.reshape(8, 8, 1),
        specularity=transmission_output.spec_map.reshape(8, 8, 1),
    )
    expected_warmup = (
        transmission_weights.diffuse
        * transmission_output.interface.render.permute(1, 2, 0)
        + transmission_weights.transmission * transmission_output.transmission_trace.rgb
    )
    assert torch.allclose(
        transmission_output.render.permute(1, 2, 0), expected_warmup, atol=1e-6
    )

    output = render_glint_transport(
        camera,
        checkpoint,
        TorchSurfelTracer(max_ray_gaussian_pairs=10_000),
    )
    assert output.render.shape == (3, 8, 8)
    assert output.material_trans_map.shape == output.trans_map.shape
    valid_coverage = output.acc_map > 1e-4
    assert torch.allclose(
        output.material_trans_map[valid_coverage] * output.acc_map[valid_coverage],
        output.trans_map[valid_coverage],
        atol=2e-5,
    )
    assert output.secondary_reflection_trace is not None
    assert output.secondary_ref_render.shape == output.ref_render.shape
    assert torch.isfinite(output.render).all()
    assert torch.allclose(
        output.energy_sum, torch.ones_like(output.energy_sum), atol=1e-6
    )

    output.render.mean().backward()
    for parameter in (
        pcd._features_dc,
        pcd._specular,
        pcd._transmission_coeff,
        reflection._features_dc,
        transmission._features_dc,
        transmission._specular,
    ):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
