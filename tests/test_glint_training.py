# SPDX-FileCopyrightText: Copyright (c) 2026 GLINT Authors
# SPDX-License-Identifier: MIT

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch

from glint.checkpoint import load_glint_checkpoint
from glint.model import (
    GlintStageSchedule,
    GlintTrainingModel,
    create_gaussian_set,
)
from glint.refinement import GlintRefiner, RefinementConfig
from glint.train import _refinement_config
from glint.trainer import (
    GlintLossConfig,
    GlintTrainer,
    GlintTrainerConfig,
    _INTERFACE_GEOMETRY_PARAMETERS,
    _binary_entropy,
    _confidence_aware_transparency_regularization,
    _depth_discrepancy_confidence,
    _disjoint_transparency_masks,
    _make_optimizers,
    _normal_agreement,
)


def _training_model(count: int = 8) -> GlintTrainingModel:
    generator = torch.Generator().manual_seed(7)
    xyz = torch.randn(count, 3, generator=generator)
    rgb = torch.rand(count, 3, generator=generator)
    return GlintTrainingModel(
        create_gaussian_set(xyz, rgb, role="interface", device="cpu"),
        create_gaussian_set(xyz, rgb, role="transmission", device="cpu"),
        create_gaussian_set(xyz, rgb, role="reflection", device="cpu"),
    )


def test_stage_schedule_boundaries_and_interface_freeze() -> None:
    schedule = GlintStageSchedule(
        transmission_start=1_000,
        reflection_start=3_000,
        interface_freeze=4_000,
    )
    assert schedule.stage(999) == "interface"
    assert schedule.stage(1_000) == "transmission"
    assert schedule.stage(2_999) == "transmission"
    assert schedule.stage(3_000) == "full"
    assert schedule.train_interface(3_999)
    assert not schedule.train_interface(4_000)


def test_three_sets_use_lean_role_specific_parameters() -> None:
    model = _training_model()
    base = {"means", "sh0", "shN", "scales", "quats", "opacities"}
    assert set(model.interface.parameter_map()) == base | {
        "specular",
        "transparency",
    }
    assert set(model.transmission.parameter_map()) == base | {"specular"}
    assert set(model.reflection.parameter_map()) == base
    assert model.interface._xyz.data_ptr() != model.transmission._xyz.data_ptr()


def test_lean_training_checkpoint_roundtrip(tmp_path: Path) -> None:
    model = _training_model()
    path = tmp_path / "training.pt"
    torch.save({"model": model.legacy_state_dict(), "epoch": 12}, path)
    checkpoint = load_glint_checkpoint(path, trainable=True)
    assert checkpoint.epoch == 12
    assert checkpoint.pcd is not None
    assert checkpoint.trans_env is not None
    assert checkpoint.env is not None
    assert set(checkpoint.pcd.parameter_map()) == set(model.interface.parameter_map())
    assert set(checkpoint.trans_env.parameter_map()) == set(
        model.transmission.parameter_map()
    )
    assert set(checkpoint.env.parameter_map()) == set(model.reflection.parameter_map())


def test_old_lean_checkpoint_restores_transmission_specularity(tmp_path: Path) -> None:
    model = _training_model()
    state = model.legacy_state_dict()
    state.pop("sampler.trans_env._specular")
    path = tmp_path / "old-lean.pt"
    torch.save({"model": state}, path)
    checkpoint = load_glint_checkpoint(path, gaussian_sets=("trans_env",))
    assert checkpoint.trans_env is not None
    assert torch.allclose(
        checkpoint.trans_env.get_specular,
        torch.full_like(checkpoint.trans_env.get_specular, 1e-3),
    )


def test_refiner_respects_capacity_and_rebinds_optimizer_parameters() -> None:
    model = _training_model(count=8)
    optimizers = _make_optimizers(model, scene_scale=1.0)["transmission"]
    refiner = GlintRefiner(
        model.transmission,
        optimizers,
        RefinementConfig(
            start=0,
            stop=10,
            every=1,
            reset_every=100,
            grow_gradient=0.1,
            grow_scale=100.0,
            max_gaussians=10,
        ),
        scene_scale=1.0,
    )
    refiner.grad_accum = torch.ones(8)
    refiner.count = torch.ones(8)
    stats = refiner.step(1)
    assert stats["duplicated"] == 2
    assert len(model.transmission.get_xyz) == 10
    for name, parameter in model.transmission.parameter_map().items():
        assert optimizers[name].param_groups[0]["params"] == [parameter]


def test_reflection_refiner_uses_relative_floor_and_adaptive_growth() -> None:
    model = _training_model(count=8)
    optimizers = _make_optimizers(model, scene_scale=1.0)["reflection"]
    refiner = GlintRefiner(
        model.reflection,
        optimizers,
        RefinementConfig(
            start=0,
            stop=10,
            every=1,
            reset_every=100,
            grow_gradient=1e9,
            grow_scale=100.0,
            max_gaussians=10,
            min_gaussian_ratio=0.5,
            adaptive_grow_quantile=0.75,
        ),
        scene_scale=1.0,
    )
    refiner.grad_accum = torch.arange(8, dtype=torch.float32)
    refiner.count = torch.ones(8)
    stats = refiner.step(1)
    assert stats["duplicated"] == 2
    assert stats["minimum"] == 4
    assert stats["grow_threshold"] < 1e9
    assert len(model.reflection.get_xyz) == 10


def test_refiner_prunes_only_low_weight_oversized_gaussians() -> None:
    model = _training_model(count=8)
    optimizers = _make_optimizers(model, scene_scale=1.0)["reflection"]
    model.reflection.parameter_map()["scales"].data.fill_(0.0)
    refiner = GlintRefiner(
        model.reflection,
        optimizers,
        RefinementConfig(
            start=0,
            stop=10,
            every=1,
            reset_every=1,
            grow_gradient=1e9,
            prune_scale=0.1,
            large_gaussian_weight_quantile=0.5,
        ),
        scene_scale=1.0,
    )
    refiner.grad_accum = torch.zeros(8)
    refiner.count = torch.ones(8)
    refiner.weight_accum = torch.arange(8, dtype=torch.float32)
    stats = refiner.step(2)
    assert stats["pruned"] == 4
    assert len(model.reflection.get_xyz) == 4


def test_refiner_clamps_scale_relative_to_scene_after_topology_window() -> None:
    model = _training_model(count=8)
    optimizers = _make_optimizers(model, scene_scale=3.0)["reflection"]
    model.reflection.parameter_map()["scales"].data.fill_(torch.tensor(10.0).log())
    refiner = GlintRefiner(
        model.reflection,
        optimizers,
        RefinementConfig(stop=0, max_scale_ratio=2.0),
        scene_scale=3.0,
    )
    stats = refiner.step(100)
    assert stats["scale_clamped"] == 8
    assert model.reflection.get_scaling.max() <= 6.0
    assert stats["max_tangent_scale_ratio"] <= 2.0 + 1e-6
    assert stats["max_tangent_covariance_eigenvalue_ratio"] <= 4.0 + 1e-6


def test_interface_covariance_projection_recovers_nonfinite_scales_and_momentum() -> None:
    model = _training_model(count=8)
    optimizers = _make_optimizers(model, scene_scale=3.0)["interface"]
    parameter = model.interface.parameter_map()["scales"]
    parameter.data.fill_(torch.tensor(0.05).log())
    parameter.sum().backward()
    optimizers["scales"].step()
    state = optimizers["scales"].state[parameter]
    state["exp_avg"].fill_(1.0)
    state["exp_avg_sq"].fill_(1.0)

    projected = torch.zeros_like(parameter, dtype=torch.bool)
    parameter.data[0, 0] = torch.tensor(1.0).log()
    parameter.data[1, 0] = float("nan")
    parameter.data[2, 1] = float("inf")
    projected[0, 0] = projected[1, 0] = projected[2, 1] = True
    refiner = GlintRefiner(
        model.interface,
        optimizers,
        RefinementConfig(stop=0, max_scale_ratio=0.1),
        scene_scale=3.0,
    )
    stats = refiner.step(100)

    assert stats["scale_clamped"] == 3
    assert stats["scale_components_clamped"] == 3
    assert stats["scale_momenta_cleared"] == 3
    assert torch.isfinite(model.interface.get_scaling).all()
    assert model.interface.get_scaling.max() <= 0.3 + 1e-6
    assert stats["max_tangent_covariance_eigenvalue_ratio"] <= 0.01 + 1e-6
    assert torch.count_nonzero(state["exp_avg"][projected]) == 0
    assert torch.count_nonzero(state["exp_avg_sq"][projected]) == 0
    assert torch.all(state["exp_avg"][~projected] == 1.0)
    assert torch.all(state["exp_avg_sq"][~projected] == 1.0)


def test_interface_covariance_cap_reuses_scene_relative_pruning_threshold() -> None:
    config = _refinement_config(
        {"max_scene_threshold": 0.07},
        prefix="",
        disabled=False,
        max_gaussians=100,
        default_every=100,
        default_reset_every=3_000,
    )
    assert config.prune_scale == 0.07
    assert config.max_scale_ratio == 0.07

    overridden = _refinement_config(
        {"max_scene_threshold": 0.07, "max_scale_ratio": 0.04},
        prefix="",
        disabled=False,
        max_gaussians=100,
        default_every=100,
        default_reset_every=3_000,
    )
    assert overridden.max_scale_ratio == 0.04


def test_loss_config_reads_original_glint_stage_windows() -> None:
    config = {
        "model_cfg": {
            "supervisor_cfg": {
                "use_normal_type": "stable",
                "norm_loss_start_iter": 10,
                "norm_loss_until_iter": 20,
                "dpt_loss_start_iter": 30,
                "dpt_loss_until_iter": None,
                "normal_cos_threshold_iter": 100,
                "normal_cos_threshold_initial": 0.4,
                "normal_cos_threshold_final": 0.8,
                "normal_cos_threshold_final_iter": 300,
                "use_normal_threshold_for_depth_loss": False,
                "trans_cleanup_start_iter": 200,
                "trans_cleanup_final_iter": 400,
                "trans_cleanup_opaque_multiplier": 2.5,
                "trans_primitive_entropy_weight": 0.7,
                "trans_cleanup_transparent_min": 0.75,
                "trans_cleanup_transparent_weight": 1.5,
                "trans_env_opacity_loss_until_iter": 40,
                "plausibility_loss_weight": 0.2,
                "perc_loss_start_iter": 45_000,
                "multi_view_loss_weight": 0.03,
            },
            "sampler_cfg": {"multi_view_every_n_iter": 7},
        },
    }
    loss = GlintLossConfig.from_easyvolcap(config)
    assert loss.normal_source == "stable"
    assert (loss.normal_prior_start, loss.normal_prior_stop) == (10, 20)
    assert (loss.depth_prior_start, loss.depth_prior_stop) == (30, None)
    assert loss.normal_cos_threshold_step == 100
    assert loss.normal_cos_threshold_initial == 0.4
    assert loss.normal_cos_threshold_final == 0.8
    assert loss.normal_cos_threshold_final_step == 300
    assert not loss.depth_normal_thresholding
    assert loss.transparency_cleanup_start == 200
    assert loss.transparency_cleanup_final_step == 400
    assert loss.transparency_cleanup_opaque_multiplier == 2.5
    assert loss.transparency_primitive_entropy_weight == 0.7
    assert loss.transparency_cleanup_transparent_min == 0.75
    assert loss.transparency_cleanup_transparent_weight == 1.5
    assert loss.transmission_opacity_stop == 40
    assert loss.plausibility == 0.2
    assert loss.perceptual_start == 45_000
    assert loss.multi_view == 0.03
    assert loss.multi_view_every == 7


def test_normal_cosine_threshold_becomes_stricter_late_in_training() -> None:
    config = GlintLossConfig(
        normal_cos_threshold_step=10,
        normal_cos_threshold_initial=0.5,
        normal_cos_threshold_final=0.9,
        normal_cos_threshold_final_step=30,
    )
    assert config.normal_cosine_threshold(9) is None
    assert config.normal_cosine_threshold(10) == 0.5
    assert config.normal_cosine_threshold(20) == 0.7
    assert config.normal_cosine_threshold(30) == 0.9
    assert config.normal_cosine_threshold(60) == 0.9


def test_normal_agreement_mask_tracks_scheduled_confidence() -> None:
    prediction = torch.tensor(
        [
            [
                [
                    [1.0, 0.0, 0.0],
                    [0.8, 0.6, 0.0],
                    [0.0, 1.0, 0.0],
                    [float("nan"), 0.0, 0.0],
                ]
            ]
        ]
    )
    target = torch.tensor(
        [[[[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]]]
    )
    cosine, early = _normal_agreement(prediction, target, 0.5)
    _, late = _normal_agreement(prediction, target, 0.9)
    assert torch.allclose(cosine, torch.tensor([[[1.0, 0.8, 0.0, 0.0]]]))
    assert early.tolist() == [[[True, True, False, False]]]
    assert late.tolist() == [[[True, False, False, False]]]


def test_scheduled_normal_agreement_mask_is_backward_safe() -> None:
    prediction = torch.tensor(
        [[[[1.0, 0.0, 0.0], [0.8, 0.6, 0.0]]]],
        requires_grad=True,
    )
    target = torch.tensor([[[[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]]])
    cosine, valid = _normal_agreement(prediction, target, 0.9)
    loss = ((1.0 - cosine) * valid.to(cosine.dtype)).sum()
    loss.backward()
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()


def test_depth_discrepancy_guidance_has_soft_dead_band_and_edge_rejection() -> None:
    config = GlintLossConfig()
    interface = torch.tensor([[[[1.0], [1.0], [2.0], [2.0]]]])
    direct = torch.tensor([[[[1.0], [1.03], [2.0], [2.06]]]])
    transparent, opaque, edge = _depth_discrepancy_confidence(
        interface,
        direct,
        config,
    )
    assert transparent[0, 0, 1, 0] > transparent[0, 0, 0, 0]
    assert transparent[0, 0, 3, 0] > transparent[0, 0, 2, 0]
    assert opaque[0, 0, 0, 0] > opaque[0, 0, 1, 0]
    assert edge[0, 0, 1, 0] < 1.0
    assert edge[0, 0, 2, 0] < 1.0


def test_transparency_cleanup_schedule_starts_after_topology_window() -> None:
    config = GlintLossConfig(
        transparency_cleanup_start=10,
        transparency_cleanup_final_step=30,
    )
    assert config.transparency_cleanup_progress(9) == 0.0
    assert config.transparency_cleanup_progress(10) == 0.0
    assert config.transparency_cleanup_progress(20) == 0.5
    assert config.transparency_cleanup_progress(30) == 1.0
    assert config.transparency_cleanup_progress(60) == 1.0


def test_transparency_guidance_discards_conflicting_soft_labels() -> None:
    transparent = torch.tensor([[[[0.8], [0.2]]]])
    opaque = torch.tensor([[[[0.3], [0.5]]]])
    resolved_transparent, resolved_opaque = _disjoint_transparency_masks(
        transparent,
        opaque,
    )
    assert torch.allclose(
        resolved_transparent,
        torch.tensor([[[[0.5], [0.0]]]]),
    )
    assert torch.allclose(
        resolved_opaque,
        torch.tensor([[[[0.0], [0.3]]]]),
    )


def test_confidence_aware_cleanup_suppresses_only_opaque_and_protects_glass() -> None:
    config = GlintLossConfig(
        transparency_cleanup_start=10,
        transparency_cleanup_final_step=20,
        transparency_cleanup_opaque_multiplier=3.0,
        transparency_cleanup_transparent_min=0.8,
        transparency_cleanup_transparent_weight=1.0,
    )
    transparency = torch.full((1, 1, 3, 1), 0.5, requires_grad=True)
    target = torch.zeros((1, 1, 3, 3))
    transparent_mask = torch.tensor([[[[1.0], [0.0], [0.0]]]])
    opaque_mask = torch.tensor([[[[0.0], [1.0], [0.0]]]])

    loss, diagnostics = _confidence_aware_transparency_regularization(
        transparency,
        target,
        transparent_mask,
        opaque_mask,
        config,
        step=20,
    )
    loss.backward()

    gradient = transparency.grad.reshape(-1)
    assert gradient[0] < 0  # gradient descent raises confident glass transparency
    assert gradient[1] > 0  # gradient descent suppresses confident opaque leakage
    assert gradient[2] == 0  # unknown pixels receive no semantic sparsity gradient
    assert diagnostics["transparency_cleanup_progress"] == 1.0
    assert diagnostics["transparency_cleanup_opaque_multiplier"] == 3.0


def test_confidence_aware_cleanup_has_no_global_transparency_mean() -> None:
    config = GlintLossConfig()
    transparency = torch.full((1, 2, 2, 1), 0.7, requires_grad=True)
    target = torch.zeros((1, 2, 2, 3))
    loss, _ = _confidence_aware_transparency_regularization(
        transparency,
        target,
        None,
        None,
        config,
        step=31_000,
    )
    loss.backward()
    assert loss == 0
    assert torch.count_nonzero(transparency.grad) == 0


def test_late_primitive_entropy_is_visible_symmetric_and_geometry_free() -> None:
    config = GlintLossConfig(
        transparency_cleanup_start=0,
        transparency_cleanup_final_step=1,
    )
    transparency = torch.full((1, 2, 2, 1), 0.5, requires_grad=True)
    primitive = torch.tensor([[0.01], [0.2], [0.99]], requires_grad=True)
    primitive_weight = torch.tensor([[1.0], [0.0], [1.0]])
    target = torch.zeros((1, 2, 2, 3))
    loss, diagnostics = _confidence_aware_transparency_regularization(
        transparency,
        target,
        None,
        None,
        config,
        step=1,
        primitive_transparency=primitive,
        primitive_weight=primitive_weight,
    )
    loss.backward()

    assert torch.count_nonzero(transparency.grad) == 0
    assert primitive.grad[0] > 0
    assert primitive.grad[1] == 0
    assert primitive.grad[2] < 0
    expected = _binary_entropy(primitive.detach()[[0, 2]]).mean()
    assert torch.allclose(diagnostics["transparency_primitive_entropy"], expected)


def test_geometry_only_freeze_keeps_interface_material_trainable() -> None:
    model = _training_model()
    trainer = SimpleNamespace(
        config=GlintTrainerConfig(interface_geometry_freeze_step=10),
        model=model,
        _interface_geometry_frozen=False,
    )
    GlintTrainer._freeze_interface_geometry_if_needed(trainer, 9)
    assert not trainer._interface_geometry_frozen
    GlintTrainer._freeze_interface_geometry_if_needed(trainer, 10)
    assert trainer._interface_geometry_frozen

    parameters = model.interface.parameter_map()
    for name in _INTERFACE_GEOMETRY_PARAMETERS:
        assert not parameters[name].requires_grad
    for name in {"sh0", "shN", "specular", "transparency"}:
        assert parameters[name].requires_grad


def test_guidance_masks_are_reused_by_cleanup_with_finite_backward() -> None:
    height, width = 2, 3
    config = GlintLossConfig(
        transparency_mask_shrink=1,
        transparency_angle_weighting=False,
        transparency_explainability_gating=False,
        transparency_cleanup_start=0,
        transparency_cleanup_final_step=1,
    )
    transparency = torch.full(
        (1, height, width, 1),
        0.5,
        requires_grad=True,
    )
    interface_depth = torch.ones(height * width, 1)
    direct_depth = torch.ones(1, height, width)
    direct_depth[:, :, :2] = 1.03
    output = SimpleNamespace(
        transmission_direct=SimpleNamespace(surf_depth=direct_depth),
        dpt_map=interface_depth,
        acc_map=torch.full((1, height * width, 1), 0.5),
        norm_map=torch.tensor([[0.0, 0.0, -1.0]]).repeat(height * width, 1),
        ray_d=torch.tensor([[0.0, 0.0, 1.0]]).repeat(height * width, 1),
        dif_render=torch.zeros(3, height, width),
        ref_render=torch.zeros(3, height, width),
        secondary_ref_render=torch.zeros(3, height, width),
        render=torch.zeros(3, height, width),
    )
    albedo = torch.full((height, width, 3), 0.5)
    albedo[:, :2] = 0.0
    sample = SimpleNamespace(
        camera=SimpleNamespace(image_height=height, image_width=width),
        diffuse_albedo=albedo,
        basecolor=torch.ones(height, width, 3),
        sky_mask=None,
        depth=None,
    )
    target = torch.zeros(1, height, width, 3)
    trainer = SimpleNamespace(loss_config=config)

    guidance = GlintTrainer._transparency_guidance(
        trainer,
        output,
        sample,
        transparency,
        target,
        step=1,
    )
    regularization, _ = _confidence_aware_transparency_regularization(
        transparency,
        target,
        output.transparency_positive_mask,
        output.transparency_negative_mask,
        config,
        step=1,
    )
    total = guidance + regularization
    total.backward()

    assert output.transparency_positive_mask[..., :2, :].sum() > 0
    assert output.transparency_negative_mask[..., 2:, :].sum() > 0
    assert torch.allclose(
        output.trans_guidance_coverage_weight,
        torch.full_like(output.trans_guidance_coverage_weight, 0.5),
    )
    assert transparency.grad is not None
    assert torch.isfinite(transparency.grad).all()


def test_refiner_opacity_reset_replaces_parameter_and_clears_adam_state() -> None:
    model = _training_model(count=8)
    optimizers = _make_optimizers(model, scene_scale=1.0)["interface"]
    opacity = model.interface.parameter_map()["opacities"]
    opacity.sum().backward()
    optimizers["opacities"].step()
    old_pointer = opacity.data_ptr()
    state = optimizers["opacities"].state[opacity]
    state["exp_avg"].fill_(1.0)
    state["exp_avg_sq"].fill_(1.0)

    refiner = GlintRefiner(
        model.interface,
        optimizers,
        RefinementConfig(
            start=0,
            stop=10,
            every=100,
            reset_every=1,
            min_gaussians=1,
        ),
        scene_scale=1.0,
    )
    refiner.step(1)

    reset = model.interface.parameter_map()["opacities"]
    assert reset.data_ptr() != old_pointer
    assert torch.allclose(torch.sigmoid(reset), torch.full_like(reset, 0.01))
    reset_state = optimizers["opacities"].state[reset]
    assert torch.count_nonzero(reset_state["exp_avg"]) == 0
    assert torch.count_nonzero(reset_state["exp_avg_sq"]) == 0


def test_refiner_never_prunes_below_configured_role_floor() -> None:
    model = _training_model(count=8)
    optimizers = _make_optimizers(model, scene_scale=1.0)["reflection"]
    model.reflection.parameter_map()["opacities"].data.fill_(
        torch.logit(torch.tensor(0.01))
    )
    refiner = GlintRefiner(
        model.reflection,
        optimizers,
        RefinementConfig(
            start=0,
            stop=10,
            every=1,
            reset_every=100,
            grow_gradient=1e9,
            prune_opacity=0.5,
            min_gaussians=4,
        ),
        scene_scale=1.0,
    )
    refiner.grad_accum = torch.zeros(8)
    refiner.count = torch.ones(8)
    stats = refiner.step(1)
    assert stats["pruned"] == 4
    assert len(model.reflection.get_xyz) == 4
