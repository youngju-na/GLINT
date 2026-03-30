# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights ...
import os
from contextlib import nullcontext
import numpy as np
from PIL import Image
from tqdm import tqdm
import argparse
import omegaconf
from omegaconf import OmegaConf
import imageio

import torch

from accelerate import Accelerator
from accelerate import PartialState
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

from src.pipelines.pipeline_rgbx import RGBXVideoDiffusionPipeline
from utils.utils_rgbx import convert_rgba_to_rgb_pil
from utils.utils_rgbx_inference import touch, find_images_recursive, base_plus_ext, \
    group_images_into_videos, split_list_with_overlap, resize_upscale_without_padding

# ---------------------------
# Helpers for uniform-cover-all
# ---------------------------
def _make_phase_buckets(F: int, N: int):
    """N개의 phase 버킷: k, k+N, k+2N, ... (빈 버킷 제외)"""
    buckets = []
    for k in range(N):
        b = list(range(k, F, N))
        if b:
            buckets.append((k, b))  # (phase_k, [orig_indices])
    return buckets

def _pad_to_len(idx_list, N):
    """길이 N이 되도록 마지막값 반복 pad"""
    if not idx_list:
        return [0] * N
    out = list(idx_list)
    while len(out) < N:
        out.append(out[-1])
    return out[:N]

def _save_input_dump_if_needed(cfg, video_save_dir, phase_k, t, orig_idx, pil_img):
    if cfg.get('save_image', False) and cfg.get('save_input_dump', False):
        save_path = os.path.join(
            video_save_dir, f"phase_{phase_k:02d}.{t:04d}.frame_{orig_idx:04d}.rgb.png"
        )
        pil_img.save(save_path)

# ---------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args, unknown = parser.parse_known_args()
    cfg = OmegaConf.load(args.config)
    cli = OmegaConf.from_dotlist(unknown)
    cfg = OmegaConf.merge(cfg, cli)

    assert cfg.get('save_video', False) or cfg.get('save_image', False)

    try:
        inference_height, inference_width = cfg.inference_res
    except:
        inference_height, inference_width = 512, 512
        print("[inference_height, inference_width] not provided or wrong; using 512x512")

    # 추가: 샘플링 모드 (None 또는 "uniform_cover_all")
    sampling_mode = cfg.get('sampling_mode', None)

    frame_reorder_mode = cfg.get('frame_reorder_mode', None)
    accelerator = Accelerator()

    cond_mode = cfg.get("cond_mode", "image")
    use_deterministic_mode = cfg.get("use_deterministic_mode", False)

    weight_dtype = cfg.get("weight_dtype", 'fp16')
    if weight_dtype == 'fp16':
        weight_dtype = torch.float16
    elif weight_dtype == 'fp32':
        weight_dtype = torch.float32

    missing_kwargs = {}
    missing_kwargs["cond_mode"] = cond_mode
    missing_kwargs["use_deterministic_mode"] = use_deterministic_mode
    if os.path.exists(cfg.inference_model_weights):
        model_weights_subfolders = os.listdir(cfg.inference_model_weights)
    else:
        model_weights_subfolders = []
    if "image_encoder" not in model_weights_subfolders:
        missing_kwargs["image_encoder"] = CLIPVisionModelWithProjection.from_pretrained(
            "stabilityai/stable-video-diffusion-img2vid", subfolder="image_encoder",
        )
        assert cond_mode != "image"
    if "feature_extractor" not in model_weights_subfolders:
        missing_kwargs["feature_extractor"] = CLIPImageProcessor.from_pretrained(
            "stabilityai/stable-video-diffusion-img2vid", subfolder="feature_extractor",
        )
        assert cond_mode != "image"

    pipeline = RGBXVideoDiffusionPipeline.from_pretrained(cfg.inference_model_weights, **missing_kwargs)
    distributed_state = PartialState()
    pipeline = pipeline.to(distributed_state.device)
    pipeline = pipeline.to(weight_dtype)
    pipeline.set_progress_bar_config(disable=True)

    default_n_frames = pipeline.unet.config.num_frames if pipeline.unet.config.num_frames else 14
    inference_n_frames = cfg.inference_n_frames if cfg.inference_n_frames else default_n_frames
    image_group_mode = cfg.image_group_mode if cfg.get("image_group_mode", None) else "folder"
    chunk_mode = cfg.chunk_mode if cfg.get("chunk_mode", None) else "first"
    overlap_n_frames = cfg.overlap_n_frames if cfg.get("overlap_n_frames", None) else 0
    subsample_every_n_frames = int(cfg.subsample_every_n_frames) if cfg.get("image_group_mode", None) else 1

    image_extensions = ['.png', '.jpg', '.jpeg', '.gif', '.tiff', '.bmp']
    if cfg.get("image_extensions", None):
        if not isinstance(cfg.image_extensions, (list, omegaconf.listconfig.ListConfig)):
            image_extensions = [str(cfg.image_extensions)]
        else:
            image_extensions = list(cfg.image_extensions)

    validation_image_paths = find_images_recursive(cfg.inference_input_dir, image_extensions=image_extensions)
    validation_video_list = group_images_into_videos(validation_image_paths, image_group_mode=image_group_mode,
                                                     subsample_every_n_frames=subsample_every_n_frames)

    inference_save_dir = cfg.inference_save_dir
    os.makedirs(inference_save_dir, exist_ok=True)

    success_signal_dir = os.path.join(inference_save_dir, "TMP_SUCCESS_SIGNAL")
    os.makedirs(success_signal_dir, exist_ok=True)

    with distributed_state.split_between_processes(validation_video_list) as validation_video_list:
        for i, input_image_relative_path_list in enumerate(
                tqdm(validation_video_list, desc="processing videos (image sequences): ")
        ):
            video_relative_base_name = base_plus_ext(input_image_relative_path_list[0], mode=image_group_mode)[0]
            video_save_dir = os.path.join(inference_save_dir, f"{video_relative_base_name}")
            if cfg.get('save_image', False):
                os.makedirs(video_save_dir, exist_ok=True)

            # ---------------------------
            # A) uniform cover-all 모드
            # ---------------------------
            if sampling_mode == "uniform_cover_all":
                F = len(input_image_relative_path_list)
                N = int(inference_n_frames)

                # phase 버킷 구성
                phase_buckets = _make_phase_buckets(F, N)  # [(phase_k, [orig_idx, ...]), ...]
                if len(phase_buckets) == 0:
                    continue

                for (phase_k, orig_indices) in phase_buckets:
                    # phase별 success signal (phase 단위로 스킵 판단)
                    success_signal_str = video_relative_base_name.replace("/", "--") + f".PHASE.{phase_k:04d}"
                    success_signal_path = os.path.join(success_signal_dir, success_signal_str)
                    if os.path.exists(success_signal_path):
                        print(f"Skipping phase: {success_signal_str}")
                        continue

                    # pad to N
                    padded_indices = _pad_to_len(orig_indices, N)

                    # 입력 이미지 준비
                    input_images_uint8 = []
                    width = height = None
                    for t, orig_idx in enumerate(padded_indices):
                        input_rel = input_image_relative_path_list[orig_idx]
                        input_path = os.path.join(cfg.inference_input_dir, input_rel)
                        input_image_pil = Image.open(input_path)
                        input_image_pil = convert_rgba_to_rgb_pil(input_image_pil, background_color=(0, 0, 0))

                        if t == 0:
                            width, height = input_image_pil.size
                            if width != inference_width or height != inference_height:
                                input_image_pil = resize_upscale_without_padding(input_image_pil, inference_height, inference_width)
                                width, height = input_image_pil.size
                        else:
                            if (input_image_pil.size[0] != width) or (input_image_pil.size[1] != height):
                                input_image_pil = input_image_pil.resize((width, height), resample=Image.BILINEAR)

                        # (옵션) 입력 덤프
                        _save_input_dump_if_needed(cfg, video_save_dir, phase_k, t, orig_idx, input_image_pil)

                        input_images_uint8.append(np.asarray(input_image_pil))

                    # BFHWC [0,1]
                    input_images = np.stack(input_images_uint8, axis=0)[None, ...].astype(np.float32) / 255.
                    cond_images = {"rgb": input_images}
                    cond_labels = {"rgb": "vae"}
                    if cond_mode == "image":
                        cond_images["clip_img"] = input_images[:, 0:1, ...]  # first frame only
                        cond_labels["clip_img"] = "clip"

                    viz_images_uint8 = input_images_uint8
                    for inference_pass in cfg.model_passes:
                        cond_images["input_context"] = inference_pass

                        if cfg.seed is None:
                            generator = None
                        else:
                            generator = torch.Generator(device=accelerator.device).manual_seed(cfg.seed)

                        if torch.backends.mps.is_available():
                            autocast_ctx = nullcontext()
                        else:
                            autocast_ctx = torch.autocast(accelerator.device.type, enabled=cfg.autocast)

                        with autocast_ctx:
                            inference_image_list = pipeline(
                                cond_images, cond_labels,
                                height=height, width=width,
                                num_frames=N,
                                num_inference_steps=cfg.inference_n_steps,
                                min_guidance_scale=cfg.inference_min_guidance_scale,
                                max_guidance_scale=cfg.inference_max_guidance_scale,
                                fps=cfg.get('fps', 7),
                                motion_bucket_id=cfg.get('motion_bucket_id', 127),
                                noise_aug_strength=cfg.get('cond_aug', 0),
                                generator=generator,
                                decode_chunk_size=cfg.get('decode_chunk_size', None),
                            ).frames[0]  # list of PIL images

                        # 저장: 항상 원본 프레임 번호 사용
                        if cfg.get('save_image', False):
                            for t in range(len(inference_image_list)):
                                orig_idx = padded_indices[t]
                                save_path = os.path.join(
                                    video_save_dir,
                                    f"phase_{phase_k:02d}.{t:04d}.frame_{orig_idx:04d}.{inference_pass}.png"
                                )
                                inference_image_list[t].save(save_path)

                        # 비디오 합성용 시각화
                        if cfg.get('save_video', False):
                            for t in range(len(viz_images_uint8)):
                                viz_images_uint8[t] = np.concatenate([
                                    viz_images_uint8[t],
                                    np.asarray(inference_image_list[t]),
                                ], axis=1)

                    # 비디오 저장
                    if cfg.get('save_video', False):
                        save_path = os.path.join(video_save_dir, f"{video_relative_base_name}.PHASE.{phase_k:04d}.viz.mp4")
                        imageio.mimsave(save_path, viz_images_uint8, fps=cfg.get('save_video_fps', 7), codec='h264')

                    # phase 완료 마킹
                    touch(success_signal_path)

                # 다음 비디오로
                continue

            # ---------------------------
            # B) 기존 순차(앞에서부터 40장씩 등) 모드 (역호환)
            # ---------------------------
            input_image_relative_path_chunks = split_list_with_overlap(
                input_image_relative_path_list, inference_n_frames, overlap_n_frames, chunk_mode=chunk_mode
            )
            if len(input_image_relative_path_chunks) == 0:
                continue

            for chunk_ind in range(len(input_image_relative_path_chunks)):
                success_signal_str = video_relative_base_name.replace("/", "--") + f".{chunk_ind:04d}"
                success_signal_path = os.path.join(success_signal_dir, success_signal_str)
                if os.path.exists(success_signal_path):
                    print(f"Skipping chunk: {success_signal_str}")
                    continue

                current_image_relative_path_list = input_image_relative_path_chunks[chunk_ind]
                # pad
                while len(current_image_relative_path_list) < inference_n_frames:
                    current_image_relative_path_list.append(current_image_relative_path_list[-1])

                # 입력 처리
                input_images_uint8 = []
                width = height = None
                for ind in range(inference_n_frames):
                    input_path = os.path.join(cfg.inference_input_dir, current_image_relative_path_list[ind])
                    input_image_pil = Image.open(input_path)
                    input_image_pil = convert_rgba_to_rgb_pil(input_image_pil, background_color=(0, 0, 0))

                    if ind == 0:
                        width, height = input_image_pil.size
                        if width != inference_width or height != inference_height:
                            input_image_pil = resize_upscale_without_padding(input_image_pil, inference_height, inference_width)
                            width, height = input_image_pil.size
                    else:
                        if width != input_image_pil.size[0] or height != input_image_pil.size[1]:
                            input_image_pil = input_image_pil.resize((width, height), resample=Image.BILINEAR)

                    # (옵션) 입력 저장
                    if cfg.get('save_image', False):
                        original_filename = os.path.basename(current_image_relative_path_list[ind])
                        original_basename, _ = os.path.splitext(original_filename)

                        if frame_reorder_mode == "move_first_50_to_end":
                            total_frames = len(input_image_relative_path_list)
                            current_global_index = chunk_ind * inference_n_frames + ind
                            if total_frames > 50:
                                if current_global_index < total_frames - 50:
                                    original_frame_index = current_global_index + 50
                                else:
                                    original_frame_index = current_global_index - (total_frames - 50)
                            else:
                                original_frame_index = current_global_index
                            save_path = os.path.join(
                                video_save_dir, f"{chunk_ind:04d}.frame_{original_frame_index:04d}.rgb.png"
                            )
                        else:
                            save_path = os.path.join(video_save_dir, f"{chunk_ind:04d}.{original_basename}.rgb.png")

                        input_image_pil.save(save_path)

                    input_images_uint8.append(np.asarray(input_image_pil))

                # 포맷/조건
                input_images = np.stack(input_images_uint8, axis=0)[None, ...].astype(np.float32) / 255.
                cond_images = {"rgb": input_images}
                cond_labels = {"rgb": "vae"}
                if cond_mode == "image":
                    cond_images["clip_img"] = input_images[:, 0:1, ...]
                    cond_labels["clip_img"] = "clip"

                viz_images_uint8 = input_images_uint8
                for inference_pass in cfg.model_passes:
                    cond_images["input_context"] = inference_pass

                    if cfg.seed is None:
                        generator = None
                    else:
                        generator = torch.Generator(device=accelerator.device).manual_seed(cfg.seed)

                    if torch.backends.mps.is_available():
                        autocast_ctx = nullcontext()
                    else:
                        autocast_ctx = torch.autocast(accelerator.device.type, enabled=cfg.autocast)

                    with autocast_ctx:
                        inference_image_list = pipeline(
                            cond_images, cond_labels,
                            height=height, width=width,
                            num_frames=inference_n_frames,
                            num_inference_steps=cfg.inference_n_steps,
                            min_guidance_scale=cfg.inference_min_guidance_scale,
                            max_guidance_scale=cfg.inference_max_guidance_scale,
                            fps=cfg.get('fps', 7),
                            motion_bucket_id=cfg.get('motion_bucket_id', 127),
                            noise_aug_strength=cfg.get('cond_aug', 0),
                            generator=generator,
                            decode_chunk_size=cfg.get('decode_chunk_size', None),
                        ).frames[0]

                    # 저장
                    if cfg.get('save_image', False):
                        for ind in range(len(inference_image_list)):
                            original_filename = os.path.basename(current_image_relative_path_list[ind])
                            original_basename, _ = os.path.splitext(original_filename)

                            if frame_reorder_mode == "move_first_50_to_end":
                                total_frames = len(input_image_relative_path_list)
                                current_global_index = chunk_ind * inference_n_frames + ind
                                if total_frames > 50:
                                    if current_global_index < total_frames - 50:
                                        original_frame_index = current_global_index + 50
                                    else:
                                        original_frame_index = current_global_index - (total_frames - 50)
                                else:
                                    original_frame_index = current_global_index
                                save_path = os.path.join(
                                    video_save_dir, f"{chunk_ind:04d}.frame_{original_frame_index:04d}.{inference_pass}.png"
                                )
                            else:
                                save_path = os.path.join(
                                    video_save_dir, f"{chunk_ind:04d}.{original_basename}.{inference_pass}.png"
                                )
                            inference_image_list[ind].save(save_path)

                    if cfg.get('save_video', False):
                        for ind in range(len(viz_images_uint8)):
                            viz_images_uint8[ind] = np.concatenate([
                                viz_images_uint8[ind],
                                np.asarray(inference_image_list[ind]),
                            ], axis=1)

                if cfg.get('save_video', False):
                    save_path = os.path.join(video_save_dir, f"{video_relative_base_name}.{chunk_ind:04d}.viz.mp4")
                    imageio.mimsave(save_path, viz_images_uint8, fps=cfg.get('save_video_fps', 7), codec='h264')

                touch(success_signal_path)


if __name__ == "__main__":
    main()
