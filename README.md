# GLINT: Modeling Scene-Scale Transparency via Gaussian Radiance Transport

[![arXiv](https://img.shields.io/badge/arXiv-2603.26181-b31b1b.svg)](https://arxiv.org/abs/2603.26181) [![GLINT](https://img.shields.io/badge/GLINT-Project%20Page-blue.svg)](https://youngju-na.github.io/GLINT) [![Dataset](https://img.shields.io/badge/Dataset-Download-green.svg)](https://drive.google.com/drive/folders/1NB_AuBQ5lP3pkdS9M-x9o0oqRrXP4S6a?usp=sharing)

Official code release for the paper: **GLINT: Modeling Scene-Scale Transparency via Gaussian Radiance Transport**.

[Youngju Na](https://youngju-na.github.io/)<sup>1,2,*</sup>, [Jaeseong Yun](mailto:jaeseong.yun@naverlabs.com)<sup>2</sup>, [Soohyun Ryu](mailto:soohyun.ryu@naverlabs.com)<sup>2</sup>, [Hyunsu Kim](https://blandocs.github.io/)<sup>2</sup>, [Sung-Eui Yoon](https://sgvr.kaist.ac.kr/~sungeui/)<sup>1</sup>, [Suyong Yeon](mailto:suyong.yeon@naverlabs.com)<sup>2</sup>

_<sup>1</sup>KAIST, <sup>2</sup>NAVER LABS_

## News
- **[2026-04-09]**: 🎉 Our paper has been selected as an Oral presentation at CVPR 2026.
- **[2026-03-30]**: Initial code release.

## Overview

GLINT is a method for modeling large-scale transparent and reflective scenes with Gaussian radiance transport.

## Installation

1. **Clone the repository and setup environment:**

```bash
conda create -n glint python=3.11 -y
conda activate glint
```

2. **Install PyTorch:**

Install PyTorch matching your CUDA version (see [PyTorch website](https://pytorch.org/get-started/locally/) for the correct command). Example for CUDA 11.8:

```bash
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu118
```

3. **Install dependencies:**

```bash
cat requirements.txt | sed -e '/^\s*-.*$/d' -e '/^\s*#.*$/d' -e '/^\s*$/d' | \
  awk '{split($0, a, "#"); if (length(a) > 1) print a[1]; else print $0;}' | \
  awk '{split($0, a, "@"); if (length(a) > 1) print a[2]; else print $0;}' | \
  xargs -n 1 pip install

pip install -e . --no-build-isolation --no-deps
```

4. **Install submodules:**

```bash
git submodule update --init --recursive
pip install -v submodules/diff-surfel-tracing
pip install \
  submodules/diff-surfel-rasterizations/diff-surfel-rasterization-wet \
  submodules/diff-surfel-rasterizations/diff-surfel-rasterization-wet-ch06 \
  submodules/diff-surfel-rasterizations/diff-surfel-rasterization-wet-ch08
```

## Data Preparation

The `ref-dl3dv` and `3D-FRONT-T` dataset used in our paper is available for download:

**[Download Dataset (Google Drive)](https://drive.google.com/drive/folders/1NB_AuBQ5lP3pkdS9M-x9o0oqRrXP4S6a?usp=sharing)**

GLINT expects datasets in the EasyVolcap-style format.
At minimum, each scene should provide:

```
<scene>/
├── images/
├── intri.yml
├── extri.yml
└── sparse/
```

For the G-buffer guidance, each scene also contains priors obtained from [DiffusionRenderer](https://arxiv.org/abs/2501.18590). You may also consider using other useful priors (e.g., [TransNormal](https://longxiang-ai.github.io/TransNormal/), [Video Depth Anything](https://github.com/DepthAnything/Video-Depth-Anything), etc.). Please refer to these if you want to build your custom datasets:

```text
<scene>/
├── images/
├── intri.yml
├── extri.yml
├── sparse/
├── envs/
│   └── points3D.ply
├── normals/
│   └── <view_id>/000000.jpg
└── diffrens/
    ├── normal/<view_id>/000000.png
    ├── depth/<view_id>/000000.png
    ├── diffuse_albedo/<view_id>/000000.png
    ├── basecolor/<view_id>/000000.png
    ├── roughness/<view_id>/000000.png
    └── metallic/<view_id>/000000.png
```

The `diffusion-renderer` prior maps must be placed under `diffrens/`.

For training and evaluation, `<scene>` should match the dataset directory name and the config filename in `configs/exps/glint/ref-dl3dv/`.

<details>
<summary>Scene list used in our dl3dv-10k subset (ref-dl3dv).</summary>

```text
194defaa605986166d52ae703b1d44d1a557794698386becaaa5f688f4fb026b
3712b8fdcb94128c92c2e2c30fb529851e3231cdc7c4451bc6c784f923386e93
52410f0264d14bde6acd695c637aaa274833be8afcf05ef4fd6a51176ad2dbd2
543b6607de9318e3a0c68b267a4b616fdc5849a140ba184807d5e70e567f8ec0
5454b71d612cc2b020e60bd2d8a018dc33d62b5fbd5c041b55a752480a8a97ba
6b42314a2f8a18a193826e2b58e45729453e74524078283f740b8f8d330c3d2f
b65e86833c1ae29714ce881bb9d14d3ed1256a08ab944fd9e75d6b29c674346d
b9df30d6e6078880acc88acb01872c65d337f84b9dba44a23fa29c9861d7e23b
```

</details>

If you want to prepare your own data from COLMAP outputs, see the preprocessing scripts in `scripts/preprocess/`.

## Training

Example training command:

```bash
evc-train -c configs/exps/glint/ref-dl3dv/<scene>.yaml \
  exp_name=glint/ref-dl3dv/<run_name>/<scene>
```

The default hyperparameters are defined in [`configs/models/glint.yaml`](configs/models/glint.yaml). Key parameters you may want to adjust depending on your scene:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `render_reflection_start_iter` | `3000` | Iteration to start reflection rendering |
| `render_transmission_start_iter` | `1000` | Iteration to start transmission rendering |
| `depth_discrepancy_threshold` | `0.005` | Depth discrepancy threshold (scene-scale dependent) |
| `trans_map_reg_loss_weight` | `0.01` | Transmission map regularization weight |
| `trans_guidance_loss_weight` | `0.01` | Transmission guidance loss weight |

## Evaluation

Example evaluation command:

```bash
evc-test -c configs/exps/glint/ref-dl3dv/<scene>.yaml \
  exp_name=glint/ref-dl3dv/<run_name>/<scene>
```

## Custom Rendering
Example interpolation video rendering:

```bash
bash scripts/render_interp_video.sh \
  --config configs/exps/glint/ref-dl3dv/<scene>.yaml \
  --exp_name glint/ref-dl3dv/<run_name>/<scene> \
  --cam_idx1 0 \
  --cam_idx2 8 \
  --n_frames 60
```

## Repository Structure
- `easyvolcap/` — Core framework and GLINT model implementation
- `configs/` — Model, dataset, and experiment configurations
- `scripts/` — Preprocessing, training utilities, and rendering scripts
- `submodules/` — Required custom CUDA and tracing dependencies

## Roadmap

- [x] Release source code.
- [ ] Release `3D-FRONT-T` Blender files for downstream applications.

## Acknowledgements

This codebase is built on top of [EasyVolcap](https://github.com/zju3dv/EasyVolcap) and the 2D Gaussian ray tracer from [EnvGS](https://github.com/zju3dv/EnvGS). We sincerely thank the authors and contributors of these projects.
You may also want to check out the related works listed below.

## Related Work

- [TSGS: Improving Gaussian Splatting for Transparent Surface Reconstruction via Normal and De-lighting Priors](https://github.com/longxiang-ai/TSGS)
- [TransparentGS: Fast Inverse Rendering of Transparent Objects with Gaussians](https://letianhuang.github.io/transparentgs/)
- [DiffusionRenderer: Neural Inverse and Forward Rendering with Video Diffusion Models](https://github.com/nv-tlabs/diffusion-renderer)
- [TransNormal: Dense Visual Semantics for Diffusion-based Transparent Object Normal Estimation](https://github.com/longxiang-ai/TransNormal)

## Citation

If you find this repository useful, please consider citing our paper:

```bibtex
@misc{na2026glint,
  title={GLINT: Modeling Scene-Scale Transparency via Gaussian Radiance Transport},
  author={Youngju Na and Jaeseong Yun and Soohyun Ryu and Hyunsu Kim and Sung-Eui Yoon and Suyong Yeon},
  year={2026},
  eprint={2603.26181},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2603.26181},
}
```

## License

This project is released under the [MIT License](LICENSE).
