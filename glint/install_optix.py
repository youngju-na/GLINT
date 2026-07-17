# SPDX-FileCopyrightText: Copyright (c) 2026 GLINT Authors
# SPDX-License-Identifier: MIT

"""Build GLINT's optional OptiX tracer in the active Python environment."""

from __future__ import annotations

import argparse
import os
import shutil
import site
import subprocess
import sys
from pathlib import Path


def _run(command: list[str], *, cwd: Path, env: dict[str, str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, env=env, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cuda-home",
        type=Path,
        default=os.environ.get("CUDA_HOME"),
        help="CUDA toolkit containing a compiler supported by the installed torch.",
    )
    parser.add_argument("--jobs", type=int, default=8)
    args = parser.parse_args()
    if args.cuda_home is None:
        raise SystemExit("CUDA_HOME is unset; pass --cuda-home")

    cuda_home = args.cuda_home.resolve()
    nvcc = cuda_home / "bin" / "nvcc"
    cuda_include = cuda_home / "targets" / "x86_64-linux" / "include"
    if not nvcc.exists() or not cuda_include.exists():
        raise SystemExit(f"Incomplete CUDA toolkit: {cuda_home}")

    repo_root = Path(__file__).resolve().parents[1]
    dependency = repo_root / "third_party" / "diff-surfel-tracing"
    if not (dependency / "setup.py").exists():
        raise SystemExit(
            "Missing third_party/diff-surfel-tracing; run "
            "`git submodule update --init --recursive` first."
        )

    env = os.environ.copy()
    env["CUDA_HOME"] = str(cuda_home)
    env["CUDACXX"] = str(nvcc)
    for name in ("CPATH", "CPLUS_INCLUDE_PATH"):
        previous = env.get(name)
        env[name] = str(cuda_include) + (os.pathsep + previous if previous else "")

    build_dir = dependency / "build"
    configure = [
        "cmake",
        "--fresh",
        "-S",
        str(dependency),
        "-B",
        str(build_dir),
        f"-DCMAKE_CUDA_COMPILER={nvcc}",
        f"-DCUDAToolkit_ROOT={cuda_home}",
        "-DCMAKE_BUILD_TYPE=Release",
    ]
    _run(configure, cwd=repo_root, env=env)
    _run(
        ["cmake", "--build", str(build_dir), "--parallel", str(args.jobs)],
        cwd=repo_root,
        env=env,
    )
    _run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-build-isolation",
            str(dependency),
        ],
        cwd=repo_root,
        env=env,
    )

    package_dir = Path(site.getsitepackages()[0]) / "diff_surfel_tracing"
    for name in ("forward.ptx", "backward.ptx"):
        source = build_dir / "ptx" / name
        if not source.exists():
            raise SystemExit(f"OptiX build did not produce {source}")
        shutil.copy2(source, package_dir / name)

    from diff_surfel_tracing import SurfelTracer

    SurfelTracer()
    print(f"GLINT OptiX tracer installed in {package_dir}")


if __name__ == "__main__":
    main()
