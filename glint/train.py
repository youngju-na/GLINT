# SPDX-FileCopyrightText: Copyright (c) 2026 GLINT Authors
# SPDX-License-Identifier: MIT

"""Train GLINT's interface, transmission, and reflection Gaussian sets."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping

import torch

from glint.checkpoint import load_glint_checkpoint
from glint.dataset import GlintDataset, load_easyvolcap_config
from glint.model import GlintStageSchedule, GlintTrainingModel
from glint.refinement import RefinementConfig
from glint.trainer import (
    GlintLossConfig,
    GlintTrainer,
    GlintTrainerConfig,
)
from glint.visualizer import (
    DEFAULT_VISUALIZATION_TYPES,
    GlintVisualizationConfig,
    GlintVisualizer,
    include_diagnostic_visualizations,
)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _configured_or_scene_path(
    explicit: Path | None,
    configured: str | Path | None,
    scene_root: Path,
    candidates: tuple[str, ...],
    *,
    description: str,
) -> Path:
    if explicit is not None:
        path = explicit
    elif configured is not None and Path(configured).exists():
        path = Path(configured)
    else:
        path = next(
            (
                scene_root / candidate
                for candidate in candidates
                if (scene_root / candidate).exists()
            ),
            Path(configured) if configured is not None else scene_root / candidates[0],
        )
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find {description} PLY at {path}. Pass its path explicitly."
        )
    return path.resolve()


def _refinement_config(
    sampler: Mapping[str, Any],
    *,
    prefix: str,
    disabled: bool,
    max_gaussians: int,
    default_every: int,
    default_reset_every: int,
    min_gaussians: int = 1_024,
    min_gaussian_ratio: float = 0.0,
    adaptive_grow_quantile: float | None = None,
    large_gaussian_weight_quantile: float | None = None,
    max_scale_ratio: float | None = None,
    middle_start: int | None = None,
    middle_stop: int | None = None,
    middle_every: int | None = None,
) -> RefinementConfig:
    key = f"{prefix}_" if prefix else ""
    default_opacity = 0.05 if prefix == "trans_env" else 0.08
    prune_scale = float(sampler.get(f"{key}max_scene_threshold", 0.1))
    # The released interface refiner already treats max_scene_threshold as the
    # oversized-surface boundary.  Reuse that scene-relative threshold as a
    # hard tangent-covariance bound instead of introducing a scene-specific
    # world-space constant.  Environment sets keep their wider explicit cap.
    if prefix == "" and max_scale_ratio is None:
        max_scale_ratio = prune_scale
    return RefinementConfig(
        start=int(sampler.get(f"{key}densify_from_iter", 500)),
        stop=0 if disabled else int(sampler.get(f"{key}densify_until_iter", 21_000)),
        every=int(sampler.get(f"{key}densification_interval", default_every)),
        middle_start=middle_start,
        middle_stop=middle_stop,
        middle_every=middle_every,
        reset_every=int(
            sampler.get(f"{key}opacity_reset_interval", default_reset_every)
        ),
        grow_gradient=float(sampler.get(f"{key}densify_grad_threshold", 2e-4)),
        grow_scale=float(sampler.get(f"{key}densify_size_threshold", 0.01)),
        prune_scale=prune_scale,
        prune_opacity=float(sampler.get(f"{key}min_opacity", default_opacity)),
        max_gaussians=max_gaussians,
        min_gaussians=min(min_gaussians, max_gaussians),
        min_gaussian_ratio=float(
            sampler.get(f"{key}min_gaussian_ratio", min_gaussian_ratio)
        ),
        adaptive_grow_quantile=(
            None
            if sampler.get(f"{key}adaptive_grow_quantile", adaptive_grow_quantile)
            is None
            else float(
                sampler.get(f"{key}adaptive_grow_quantile", adaptive_grow_quantile)
            )
        ),
        large_gaussian_weight_quantile=(
            None
            if sampler.get(
                f"{key}min_weight_threshold",
                large_gaussian_weight_quantile,
            )
            is None
            else float(
                sampler.get(
                    f"{key}min_weight_threshold",
                    large_gaussian_weight_quantile,
                )
            )
        ),
        max_scale_ratio=(
            None
            if sampler.get(f"{key}max_scale_ratio", max_scale_ratio) is None
            else float(sampler.get(f"{key}max_scale_ratio", max_scale_ratio))
        ),
    )


def _load_resume_model(
    path: Path, device: torch.device, trust: bool
) -> GlintTrainingModel:
    checkpoint = load_glint_checkpoint(
        path,
        device=device,
        trainable=True,
        trust_checkpoint=trust,
    )
    if checkpoint.pcd is None or checkpoint.trans_env is None or checkpoint.env is None:
        raise ValueError("A training checkpoint must contain all three Gaussian sets")
    model = GlintTrainingModel(
        checkpoint.pcd,
        checkpoint.trans_env,
        checkpoint.env,
    )
    with torch.no_grad():
        model.interface_background.copy_(checkpoint.bg_color)
        model.transmission_background.copy_(checkpoint.trans_env_bg_color)
        model.reflection_background.copy_(checkpoint.env_bg_color)
    return model


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Standalone gsplat trainer for canonical three-set GLINT."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--interface-ply", type=Path)
    parser.add_argument("--reflection-ply", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--trust-checkpoint",
        action="store_true",
        help="Allow pickle loading when resuming a trusted legacy GLINT checkpoint.",
    )
    parser.add_argument("--ratio", type=float)
    parser.add_argument("--max-steps", type=_positive_int)
    parser.add_argument("--transmission-start", type=int)
    parser.add_argument("--reflection-start", type=int)
    parser.add_argument("--interface-freeze", type=int)
    parser.add_argument(
        "--interface-geometry-freeze",
        type=int,
        help=(
            "Freeze interface means/scales/quaternions/opacities at this step "
            "while continuing to optimize color and material parameters."
        ),
    )
    parser.add_argument("--max-interface-points", type=_positive_int)
    parser.add_argument("--max-reflection-points", type=_positive_int)
    parser.add_argument("--max-gaussians", type=_positive_int, default=1_000_000)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=_positive_int)
    parser.add_argument("--save-every", type=int)
    parser.add_argument("--image-every", type=int)
    parser.add_argument(
        "--visualization-types",
        nargs="+",
        help="Override the GLINT visualization types from runner_cfg.visualizer_cfg.",
    )
    parser.add_argument("--visualization-columns", type=_positive_int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-refinement", action="store_true")
    args = parser.parse_args()

    if args.ratio is not None and args.ratio <= 0:
        parser.error("--ratio must be positive")
    if args.num_workers < 0:
        parser.error("--num-workers must be non-negative")
    if args.save_every is not None and args.save_every < 0:
        parser.error("--save-every must be non-negative")
    if args.image_every is not None and args.image_every < 0:
        parser.error("--image-every must be non-negative")
    if (
        args.interface_geometry_freeze is not None
        and args.interface_geometry_freeze < 0
    ):
        parser.error("--interface-geometry-freeze must be non-negative")

    config = load_easyvolcap_config(args.config)
    runner = config.get("runner_cfg", {})
    sampler = config.get("model_cfg", {}).get("sampler_cfg", {})
    loss_config = GlintLossConfig.from_easyvolcap(config)
    dataset_overrides: dict[str, Any] = {"load_neighbors": loss_config.multi_view > 0}
    if args.data_root is not None:
        dataset_overrides["data_root"] = args.data_root
    if args.ratio is not None:
        dataset_overrides["ratio"] = args.ratio
    dataset = GlintDataset.from_config(
        args.config,
        split="train",
        **dataset_overrides,
    )

    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("GLINT training requires CUDA for the OptiX tracing stages")
    torch.backends.cuda.matmul.allow_tf32 = bool(config.get("allow_tf32", True))

    if args.resume is not None:
        model = _load_resume_model(args.resume, device, args.trust_checkpoint)
        initialization = f"resume={args.resume.resolve()}"
    else:
        interface_ply = _configured_or_scene_path(
            args.interface_ply,
            None if args.data_root is not None else sampler.get("preload_gs"),
            dataset.data_root,
            ("sparse/0/points3D.ply", "sparse/points3D.ply"),
            description="interface/transmission initialization",
        )
        reflection_ply = _configured_or_scene_path(
            args.reflection_ply,
            None if args.data_root is not None else sampler.get("env_preload_gs"),
            dataset.data_root,
            ("envs/points3D.ply",),
            description="reflection initialization",
        )
        model = GlintTrainingModel.from_ply(
            interface_ply,
            reflection_ply,
            sh_degree=int(sampler.get("sh_deg", 3)),
            environment_sh_degree=int(sampler.get("env_sh_deg", 3)),
            max_interface_points=args.max_interface_points,
            max_reflection_points=args.max_reflection_points,
            device=device,
            seed=args.seed,
        )
        initialization = f"interface={interface_ply} reflection={reflection_ply}"

    transmission_start = (
        int(args.transmission_start)
        if args.transmission_start is not None
        else int(sampler.get("render_transmission_start_iter", 1_000))
    )
    reflection_start = (
        int(args.reflection_start)
        if args.reflection_start is not None
        else int(sampler.get("render_reflection_start_iter", 3_000))
    )
    if args.interface_freeze is not None:
        interface_freeze = args.interface_freeze
    elif bool(sampler.get("pcd_freeze_enabled", False)):
        interface_freeze = int(sampler.get("pcd_freeze_iter", 31_000))
    else:
        interface_freeze = None
    schedule = GlintStageSchedule(
        transmission_start=transmission_start,
        reflection_start=reflection_start,
        interface_freeze=interface_freeze,
    )

    max_steps = (
        args.max_steps
        if args.max_steps is not None
        else int(runner.get("epochs", 120)) * int(runner.get("ep_iter", 500))
    )
    xyz_schedule = sampler.get("xyz_lr_scheduler", {})
    trainer_config = GlintTrainerConfig(
        max_steps=max_steps,
        sh_degree_interval=int(sampler.get("sh_update_iter", 1_000)),
        interface_sh_start=int(sampler.get("sh_start_iter", 0) or 0),
        environment_sh_start=int(sampler.get("env_sh_start_iter", 0) or 0),
        position_lr_max_steps=int(xyz_schedule.get("max_steps", 30_000)),
        interface_geometry_freeze_step=args.interface_geometry_freeze,
        log_every=args.log_every or int(runner.get("log_interval", 10)),
        save_every=(5_000 if args.save_every is None else args.save_every),
        image_every=(1_000 if args.image_every is None else args.image_every),
        num_workers=args.num_workers,
        seed=args.seed,
        scene_scale=float(sampler.get("spatial_scale", 1.0)),
    )
    refinement_configs = {
        "interface": _refinement_config(
            sampler,
            prefix="",
            disabled=args.no_refinement,
            max_gaussians=args.max_gaussians,
            default_every=int(sampler.get("init_densification_interval", 100)),
            default_reset_every=3_000,
            middle_start=reflection_start,
            middle_stop=int(sampler.get("normal_prop_until_iter", 18_000)),
            middle_every=int(sampler.get("norm_densification_interval", 500)),
        ),
        "transmission": _refinement_config(
            sampler,
            prefix="trans_env",
            disabled=args.no_refinement,
            max_gaussians=args.max_gaussians,
            default_every=500,
            default_reset_every=6_000,
            min_gaussian_ratio=0.01,
            large_gaussian_weight_quantile=0.1,
            max_scale_ratio=2.0,
        ),
        "reflection": _refinement_config(
            sampler,
            prefix="env",
            disabled=args.no_refinement,
            max_gaussians=args.max_gaussians,
            default_every=500,
            default_reset_every=6_000,
            min_gaussian_ratio=0.05,
            adaptive_grow_quantile=0.95,
            large_gaussian_weight_quantile=0.1,
            max_scale_ratio=2.0,
        ),
    }
    configured_visualization = runner.get("visualizer_cfg", {}).get(
        "types", DEFAULT_VISUALIZATION_TYPES
    )
    visualization_types = (
        tuple(str(name).upper() for name in args.visualization_types)
        if args.visualization_types is not None
        else include_diagnostic_visualizations(configured_visualization)
    )
    visualizer = GlintVisualizer(
        GlintVisualizationConfig(
            types=visualization_types,
            columns=args.visualization_columns,
            normal_source=loss_config.normal_source,
        )
    )
    trainer = GlintTrainer(
        model,
        dataset,
        output_dir=args.output_dir,
        schedule=schedule,
        loss_config=loss_config,
        trainer_config=trainer_config,
        refinement_configs=refinement_configs,
        visualizer=visualizer,
    )
    if args.resume is not None:
        trainer.restore_training_state(
            args.resume,
            legacy_steps_per_epoch=int(runner.get("ep_iter", 500)),
        )
        if trainer.step >= max_steps:
            raise ValueError(
                f"Resume starts at step {trainer.step}, but max_steps is {max_steps}"
            )

    parameter_schema = {
        name: list(gaussian_set.parameter_map())
        for name, gaussian_set in (
            ("interface", model.interface),
            ("transmission", model.transmission),
            ("reflection", model.reflection),
        )
    }
    counts = {
        "interface": len(model.interface.get_xyz),
        "transmission": len(model.transmission.get_xyz),
        "reflection": len(model.reflection.get_xyz),
    }
    print(f"dataset={dataset.summary()}")
    print(f"initialization={initialization}")
    print(f"schedule={schedule} max_steps={max_steps}")
    print(f"gaussians={counts} parameters={parameter_schema}")
    trainer.train()


if __name__ == "__main__":
    main()
