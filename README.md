

# GLINT: Modeling Scene-Scale Transparency via Gaussian Radiance Transport

[![arXiv](https://img.shields.io/badge/arXiv-2603.26181-b31b1b.svg)](https://arxiv.org/abs/2603.26181)
[![Project page](https://img.shields.io/badge/GLINT-Project%20Page-blue.svg)](https://youngju-na.github.io/GLINT)
[![Dataset](https://img.shields.io/badge/Dataset-Download-green.svg)](https://drive.google.com/drive/folders/1NB_AuBQ5lP3pkdS9M-x9o0oqRrXP4S6a?usp=sharing)

Official implementation of **GLINT: Modeling Scene-Scale Transparency via
Gaussian Radiance Transport**.

[Youngju Na](https://youngju-na.github.io/)<sup>1,2,*</sup>,
[Jaeseong Yun](mailto:jaeseong.yun@naverlabs.com)<sup>2</sup>,
[Soohyun Ryu](mailto:soohyun.ryu@naverlabs.com)<sup>2</sup>,
[Hyunsu Kim](https://blandocs.github.io/)<sup>2</sup>,
[Sung-Eui Yoon](https://sgvr.kaist.ac.kr/~sungeui/)<sup>1</sup>,
[Suyong Yeon](mailto:suyong.yeon@naverlabs.com)<sup>2</sup>

_<sup>1</sup>KAIST, <sup>2</sup>NAVER LABS_

> [!NOTE]
> The `main` branch contains the new
> [gsplat](https://github.com/nerfstudio-project/gsplat)-based implementation.
> The original EasyVolCap-based release is preserved on the
> [`easyvolcap`](https://github.com/youngju-na/GLINT/tree/easyvolcap) branch.

## News

- **[2026-07-16]**: Released the
  [gsplat](https://github.com/nerfstudio-project/gsplat)-based implementation of
  GLINT. The original implementation remains available on the `easyvolcap`
  branch.
- **[2026-06-07]**: Updates with bug fixes and minor improvements are coming soon.
- **[2026-06-07]**: 🎉 Our paper has been selected as an Award Candidate!
- **[2026-04-09]**: 🎉 Our paper has been selected for an Oral presentation at CVPR 2026.
- **[2026-03-30]**: Initial code release.

## Installation

We tested the code on Linux with Python 3.11, PyTorch 2.9.1+cu126, the CUDA
12.9 toolkit, and an NVIDIA RTX 4090.

Clone recursively so that the OptiX tracer dependency is present:

```bash
git clone --recursive https://github.com/youngju-na/GLINT.git
cd GLINT
```

Create a separate environment for the gsplat implementation:

```bash
conda create -n splat python=3.11 cmake ninja -y
conda activate splat
```

Install PyTorch for your CUDA runtime. The tested combination is:

```bash
pip install torch==2.9.1 torchvision==0.24.1 \
  --index-url https://download.pytorch.org/whl/cu126
```

Compiling gsplat and the OptiX tracer requires a CUDA toolkit with `nvcc` and
the CUDA development headers. Set `CUDA_HOME` before installing them. On
toolkits that place headers under `targets/x86_64-linux`, also expose that
directory to the host compiler:

```bash
export CUDA_HOME=/path/to/cuda
export CPATH="${CUDA_HOME}/targets/x86_64-linux/include${CPATH:+:${CPATH}}"
export CPLUS_INCLUDE_PATH="${CUDA_HOME}/targets/x86_64-linux/include${CPLUS_INCLUDE_PATH:+:${CPLUS_INCLUDE_PATH}}"
```

Install the GLINT-compatible gsplat commit separately. gsplat is an external
dependency rather than a copy embedded in this repository:

```bash
BUILD_2DGS=1 NUM_CHANNELS=3,4,6,7 BUILD_EXPERIMENTAL=0 \
  pip install --no-build-isolation \
  "git+https://github.com/youngju-na/gsplat.git@83a0dcd3f850cccdc7b92b50fd6a568e3260eae1"
```

GLINT uses gsplat's 2DGS rasterizer; the other gsplat kernels are not required.

Then install the remaining Python packages and GLINT:

```bash
pip install -r requirements.txt
pip install --no-deps -e .
```

Build the differentiable OptiX tracer using the same CUDA toolkit:

```bash
python -m glint.install_optix
```

Check the installation and run the tests:

```bash
python -m glint.verify_installation
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q tests/test_glint_*.py
```

The PyTorch reference tracer is intended only for unit tests and tiny scenes.
Full-resolution training requires the OptiX backend.

## Data preparation

The Ref-DL3DV and 3D-FRONT-T datasets used by GLINT are available from the
[dataset download](https://drive.google.com/drive/folders/1NB_AuBQ5lP3pkdS9M-x9o0oqRrXP4S6a?usp=sharing).
Each extracted scene follows the released EasyVolCap-style layout:

```text
<scene>/
├── images/
├── intri.yml
├── extri.yml
├── sparse/0/points3D.ply
├── envs/points3D.ply
├── normals/                    # stable-normal priors, when used
└── diffrens/
    ├── normal/
    ├── depth/
    ├── diffuse_albedo/
    ├── basecolor/
    ├── roughness/
    └── metallic/
```

`diffrens` contains the G-buffer priors used for supervision. The downloadable
datasets already include the [DiffusionRenderer](https://github.com/nv-tlabs/diffusion-renderer)
priors used in the paper, and the official configurations use
`diffrens/normal`. No additional normal estimator is required to reproduce the
released setup.

For custom data or alternative priors, normal maps can also be generated with
[StableNormal](https://github.com/Stable-X/StableNormal),
[TransNormal](https://github.com/longxiang-ai/TransNormal), or
[Metric3D](https://github.com/YvanYin/Metric3D), while
[UniRelight](https://github.com/nv-tlabs/UniRelight) provides intrinsic
decomposition and video relighting. Convert normal predictions to GLINT's
camera-space convention and store them under `normals/<camera>/<frame>` before
selecting `use_normal_type: stable`. The `normals` directory is optional;
missing stable-normal predictions are ignored by the data loader.

Official view splits and scene hyperparameters are provided under
[`configs/exps/glint`](configs/exps/glint). A local scene path can always be
provided with `--data-root`; no source or YAML edit is required.

## Training

Activate the `splat` environment and run commands from the repository root.

### Ref-DL3DV

```bash
SCENE=6b42314a2f8a18a193826e2b58e45729453e74524078283f740b8f8d330c3d2f

python -m glint.train \
  --config configs/exps/glint/ref-dl3dv/${SCENE}.yaml \
  --data-root /path/to/ref-dl3dv/${SCENE} \
  --output-dir results/glint/ref-dl3dv/${SCENE} \
  --max-steps 60000 \
  --interface-geometry-freeze 31000
```

The [`train_ref_dl3dv_6b42.sh`](scripts/train_ref_dl3dv_6b42.sh)
script provides a short validation run, training, resume, and evaluation
commands:

```bash
DATA_ROOT=/path/to/ref-dl3dv/${SCENE} \
  bash scripts/train_ref_dl3dv_6b42.sh quick

DATA_ROOT=/path/to/ref-dl3dv/${SCENE} \
  bash scripts/train_ref_dl3dv_6b42.sh train
```

### 3D-FRONT-T

```bash
SCENE=scene_4

python -m glint.train \
  --config configs/exps/glint/3d-front-t/${SCENE}.yaml \
  --data-root /path/to/3d-front-t/${SCENE} \
  --output-dir results/glint/3d-front-t/${SCENE} \
  --max-steps 60000 \
  --interface-geometry-freeze 31000
```

The ten-scene sequential benchmark wrapper accepts dataset roots through
environment variables:

```bash
REF_DL3DV_ROOT=/path/to/ref-dl3dv \
SYNTHETIC_ROOT=/path/to/3d-front-t \
  bash scripts/train_benchmark_10scenes.sh all
```

## Evaluation and rendering

Evaluate the official validation split and save component visualizations:

```bash
python -m glint.evaluate_dataset \
  --config configs/exps/glint/ref-dl3dv/${SCENE}.yaml \
  --data-root /path/to/ref-dl3dv/${SCENE} \
  --checkpoint results/glint/ref-dl3dv/${SCENE}/checkpoints/latest.pt \
  --output-dir results/glint/ref-dl3dv/${SCENE}/eval \
  --save-images \
  --save-visualizations
```

### Geometry evaluation

The 3D-FRONT-T geometry benchmark reports normal MAE and angular accuracy,
depth AbsRel, normalized RMSE and δ < 1.25, and mesh Chamfer distance and F1.
First render the validation views with raw geometry output enabled:

```bash
SCENE=scene_4

python -m glint.evaluate_dataset \
  --config configs/exps/glint/3d-front-t/${SCENE}.yaml \
  --data-root /path/to/3d-front-t/${SCENE} \
  --checkpoint results/glint/3d-front-t/${SCENE}/checkpoints/latest.pt \
  --output-dir results/glint/3d-front-t/${SCENE}/eval \
  --save-geometry
```

Evaluate the saved z-depth and camera-space normal maps against the geometry
ground truth:

```bash
python -m glint.evaluate_geometry maps \
  --prediction-root results/glint/3d-front-t/${SCENE}/eval/geometry \
  --ground-truth-root /path/to/geometry_gt/${SCENE} \
  --output results/glint/3d-front-t/${SCENE}/eval/geometry_metrics.json
```

The evaluator applies the paper protocol: the released geometry view split,
per-view median depth alignment, macro-averaging over validation views, and
RMSE normalization by mean valid GT depth. The JSON report also includes RMSE
in the input depth units. Use `--all-views` only for a custom split.

After evaluating all five scenes, reproduce the table-level macro-average with:

```bash
python -m glint.evaluate_geometry summary \
  --reports results/glint/3d-front-t/scene_{1,2,3,4,5}/eval/geometry_metrics.json \
  --output results/glint/3d-front-t/geometry_metrics.json
```

For mesh evaluation, install the optional CPU dependencies and compare the
post-processed interface mesh produced by TSDF fusion:

```bash
pip install ".[geometry]"
```

python -m glint.evaluate_geometry mesh \
  --prediction /path/to/tsdf_fusion_interface_post.ply \
  --ground-truth /path/to/geometry_gt/${SCENE}/gt_mesh.ply \
  --output results/glint/3d-front-t/${SCENE}/eval/mesh_metrics.json
```

Chamfer distance is computed from mesh surfaces sampled at 1.5 cm spacing. The
report contains meters and decimeters; Table 1 uses decimeters. F1 uses a 1 cm
distance threshold. The geometry ground-truth package is expected in this
layout:

```text
geometry_gt/<scene>/
├── depths_gt/val_depthZ_XXXX.npy
├── normals_gt/val_normalCam_XXXX.npy
└── gt_mesh.ply
```

Training writes the following under `--output-dir`:

```text
<output>/
├── checkpoints/
│   ├── latest.pt
│   └── step_*.pt
├── train.log
└── train_visualizations/
    ├── PANELS/
    ├── RENDER/
    ├── DEPTH/
    ├── NORMAL/
    ├── TRANSPARENCY/
    └── ...
```

The visualization includes direct and transported RGB, depth, normal, alpha,
specularity, conditional material transparency, the composited transparency
gate, diffuse/transmission/reflection contributions, guidance masks, and
combined visualization panels.

## Citation

```bibtex
@inproceedings{na2026glint,
  title     = {{GLINT}: Modeling Scene-Scale Transparency via Gaussian Radiance Transport},
  author    = {Na, Youngju and Yun, Jaeseong and Ryu, Soohyun and Kim, Hyunsu and Yoon, Sung-Eui and Yeon, Suyong},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year      = {2026}
}
```

This implementation is built on [gsplat](https://github.com/nerfstudio-project/gsplat).
Please also cite gsplat when this backend is used; its citation is included in
[`CITATION.bib`](CITATION.bib).

## License and acknowledgements

GLINT is released under the MIT license in [`LICENSE`](LICENSE). gsplat is an
external dependency distributed under Apache-2.0, and recursive third-party
submodules retain their own licenses. See
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for attribution.

We thank the authors and maintainers of gsplat, EasyVolCap, 2D Gaussian
Splatting, and the differentiable surfel tracing implementation on which this
release builds.
