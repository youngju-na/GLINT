#!/usr/bin/env python3
"""
Simple FPS benchmark for EasyVolcap models.

Examples:
    python3 scripts/benchmark_fps_simple.py \
        -c configs/exps/envgs/manual_synthetic_GT_cam_points/scene_4.yaml \
        exp_name=envgs/ablations/time_report/scene_4

    python3 scripts/benchmark_fps_simple.py \
        configs/exps/envgs/manual_synthetic_GT_cam_points/scene_4.yaml \
        exp_name=envgs/ablations/time_report/scene_4 \
        --output data/result/envgs/ablations/time_report/scene_4/time_report/fps_benchmark.json
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Benchmark inference FPS with synchronized wall-clock timing.")
    parser.add_argument("-c", "--config", default="", help="Config path(s) passed to EasyVolcap.")
    parser.add_argument("--output", default="", help="Optional JSON output path.")
    parser.add_argument("--warmup-frames", type=int, default=5, help="Number of warmup frames to skip.")
    parser.add_argument("--benchmark-frames", type=int, default=50, help="Number of frames to benchmark.")
    parser.add_argument("--base-device", default="cuda", help="Device used to build the runner.")
    parser.add_argument("--print-progress", action="store_true", help="Print EasyVolcap test progress bar while building runner.")
    args, extra = parser.parse_known_args(argv)

    # Backward-compatible positional config handling:
    # `python script.py configs/foo.yaml exp_name=bar`
    if not args.config and extra and (extra[0].endswith(".yaml") or "," in extra[0]):
        args.config = extra[0]
        extra = extra[1:]

    if not args.config:
        parser.error("A config path is required. Pass it with `-c` or as the first positional argument.")

    return args, extra


def sync_if_needed(device: str):
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def main():
    os.environ["PYTHONUNBUFFERED"] = "1"

    args, easyvolcap_overrides = parse_args(sys.argv[1:])

    # EasyVolcap reads from sys.argv during import/initialization.
    sys.argv = [sys.argv[0], "-t", "test", "-c", args.config] + easyvolcap_overrides

    from easyvolcap.engine import cfg
    from easyvolcap.scripts.main import test
    from easyvolcap.utils.console_utils import blue, green, log, yellow
    from easyvolcap.utils.data_utils import to_cuda

    log(yellow("=" * 60))
    log(yellow("FPS Benchmark"))
    log(yellow("=" * 60))
    log(f"Config: {args.config}")
    log(f"Overrides: {easyvolcap_overrides}")

    runner = test(
        cfg,
        base_device=args.base_device,
        dry_run=True,
        record_images_to_tb=False,
        print_test_progress=args.print_progress,
    )
    runner.load_network()
    runner.model.eval()

    val_dataloader = runner.val_dataloader
    total_frames = len(val_dataloader)
    warmup_frames = min(args.warmup_frames, total_frames)
    benchmark_frames = min(args.benchmark_frames, max(total_frames - warmup_frames, 0))

    if benchmark_frames <= 0:
        raise RuntimeError(
            f"Not enough validation frames to benchmark: total={total_frames}, warmup={warmup_frames}, "
            f"benchmark={benchmark_frames}"
        )

    log(f"Available frames: {total_frames}")
    log(f"Warmup frames:    {warmup_frames}")
    log(f"Benchmark frames: {benchmark_frames}")

    with torch.no_grad():
        for i, batch in enumerate(val_dataloader):
            if i >= warmup_frames:
                break
            batch = to_cuda(batch)
            _ = runner.model(batch)

    frame_times = []

    with torch.no_grad():
        sync_if_needed(args.base_device)
        total_start = time.perf_counter()

        for i, batch in enumerate(val_dataloader):
            if i < warmup_frames:
                continue
            if i >= warmup_frames + benchmark_frames:
                break

            batch = to_cuda(batch)

            sync_if_needed(args.base_device)
            frame_start = time.perf_counter()
            _ = runner.model(batch)
            sync_if_needed(args.base_device)
            frame_end = time.perf_counter()

            frame_times.append(frame_end - frame_start)

            finished = i - warmup_frames + 1
            if finished % 10 == 0 or finished == benchmark_frames:
                log(f"Progress: {finished}/{benchmark_frames}")

        sync_if_needed(args.base_device)
        total_end = time.perf_counter()

    frame_times = np.asarray(frame_times, dtype=np.float64)
    total_time = float(total_end - total_start)
    mean_frame_time = float(frame_times.mean())
    mean_fps = float(len(frame_times) / total_time)

    results = {
        "config": args.config,
        "exp_name": cfg.exp_name,
        "base_device": args.base_device,
        "trained_model_dir": runner.trained_model,
        "total_frames_available": int(total_frames),
        "warmup_frames": int(warmup_frames),
        "benchmark_frames": int(len(frame_times)),
        "total_time_s": total_time,
        "mean_fps": mean_fps,
        "mean_frame_time_s": mean_frame_time,
        "mean_frame_time_ms": float(mean_frame_time * 1000.0),
        "std_frame_time_ms": float(frame_times.std() * 1000.0),
        "min_frame_time_ms": float(frame_times.min() * 1000.0),
        "max_frame_time_ms": float(frame_times.max() * 1000.0),
        "median_frame_time_ms": float(np.median(frame_times) * 1000.0),
        "p95_frame_time_ms": float(np.percentile(frame_times, 95) * 1000.0),
        "p99_frame_time_ms": float(np.percentile(frame_times, 99) * 1000.0),
        "overrides": easyvolcap_overrides,
    }

    output_path = args.output
    if not output_path:
        output_path = os.path.join("data", "result", cfg.exp_name, "fps_benchmark.json")

    output_path = str(Path(output_path))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    log(green("=" * 60))
    log(green("Benchmark Results"))
    log(green("=" * 60))
    log(f"Total frames:      {results['benchmark_frames']}")
    log(f"Total time:        {results['total_time_s']:.3f}s")
    log(yellow(f"Mean FPS:          {results['mean_fps']:.2f}"))
    log(f"Mean frame time:   {results['mean_frame_time_ms']:.2f}ms")
    log(f"Std frame time:    {results['std_frame_time_ms']:.2f}ms")
    log(f"Min frame time:    {results['min_frame_time_ms']:.2f}ms")
    log(f"Max frame time:    {results['max_frame_time_ms']:.2f}ms")
    log(f"Median frame time: {results['median_frame_time_ms']:.2f}ms")
    log(f"P95 frame time:    {results['p95_frame_time_ms']:.2f}ms")
    log(f"P99 frame time:    {results['p99_frame_time_ms']:.2f}ms")
    log(f"Saved JSON:        {blue(output_path)}")
    log(green("=" * 60))


if __name__ == "__main__":
    main()
