# SPDX-FileCopyrightText: Copyright (c) 2026 GLINT Authors
# SPDX-License-Identifier: MIT

"""Check the GLINT renderer installation with a small forward/backward pass."""

from __future__ import annotations

import torch

from glint.renderer import rasterize_glint_2dgs


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("The GLINT installation check requires a CUDA device")

    torch.manual_seed(7)
    device = torch.device("cuda")
    n, height, width = 32, 64, 64
    means = torch.cat(
        (
            torch.empty(n, 2, device=device).uniform_(-0.6, 0.6),
            torch.empty(n, 1, device=device).uniform_(2.0, 4.0),
        ),
        dim=-1,
    ).requires_grad_()
    quats = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).repeat(n, 1)
    quats.requires_grad_()
    scales = torch.full((n, 2), 0.18, device=device, requires_grad=True)
    opacities = torch.full((n, 1), 0.65, device=device, requires_grad=True)
    features = torch.rand(n, 5, device=device, requires_grad=True)
    viewmat = torch.eye(4, device=device)
    K = torch.tensor(
        [[50.0, 0.0, width / 2], [0.0, 50.0, height / 2], [0.0, 0.0, 1.0]],
        device=device,
    )

    output = rasterize_glint_2dgs(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        features=features,
        viewmat=viewmat,
        K=K,
        width=width,
        height=height,
        background=torch.zeros(3, device=device),
        specular_channels=1,
    )
    loss = (
        output.render.square().mean()
        + output.rend_alpha.mean()
        + output.specular.square().mean()
        + output.transparency.square().mean()
        + 0.01 * output.rend_dist.mean()
    )
    loss.backward()

    for name, value in {
        "means": means,
        "quats": quats,
        "scales": scales,
        "opacities": opacities,
        "features": features,
    }.items():
        if value.grad is None or not torch.isfinite(value.grad).all():
            raise RuntimeError(f"invalid {name} gradient")
    if output.viewspace_points.grad is None:
        raise RuntimeError("legacy viewspace gradient was not captured")

    print(f"render={tuple(output.render.shape)} alpha={tuple(output.rend_alpha.shape)}")
    print(f"visible={int(output.visibility_filter.sum())}/{n} loss={loss.item():.6f}")
    print("GLINT_INSTALLATION_CHECK=PASS")


if __name__ == "__main__":
    main()
