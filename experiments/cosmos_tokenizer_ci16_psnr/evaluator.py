#!/usr/bin/env python3
"""ASI-Evolve evaluator for Cosmos CI16 tokenizer Stage 2 config search."""

from __future__ import annotations

import argparse
import copy
import glob
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml


COSMOS_ROOT = Path("/home/cyprian/Work/projects/cosmos-tokenizer")
TRAINING_ROOT = COSMOS_ROOT / "training"
BASE_CONFIG = TRAINING_ROOT / "configs" / "asi_stage2_psnr_search.yaml"
BASE_STAGE1_CHECKPOINT = (
    TRAINING_ROOT
    / "pretrained"
    / "Baseline_64"
    / "checkpoints"
    / "stage1_checkpoint_best.pt"
)
OUTPUT_ROOT = TRAINING_ROOT / "outputs_asi"
OPENPANEL_PROJECT = "cosmos-tokenizer"
PSNR_TARGET = 30.0

FIXED_MODEL = {
    "spatial_compression": 16,
    "latent_channels": 16,
    "z_channels": 16,
    "z_factor": 1,
    "patch_size": 4,
    "patch_method": "haar",
    "channels": 64,
    "channels_mult": [1, 2, 2],
    "num_res_blocks": 1,
    "attn_resolutions": [16],
    "dropout": 0.0,
    "formulation": "AE",
    "in_channels": 3,
    "out_channels": 3,
    "resolution": 64,
    "name": "CI16x16-asi-psnr-search",
}


def load_yaml_text(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    fenced = re.search(r"```(?:yaml|yml)?\s*(.*?)```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)
    data = yaml.safe_load(text) or {}
    if not isinstance(data, dict):
        raise ValueError("candidate must be a YAML mapping")
    return data


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if (
            isinstance(value, dict)
            and isinstance(result.get(key), dict)
        ):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def clamp_number(
    root: dict[str, Any],
    path: tuple[str, ...],
    low: float,
    high: float,
    *,
    as_int: bool = False,
) -> None:
    target = root
    for key in path[:-1]:
        if key not in target or not isinstance(target[key], dict):
            return
        target = target[key]
    key = path[-1]
    if key not in target:
        return
    try:
        value = float(target[key])
    except (TypeError, ValueError):
        return
    value = max(low, min(high, value))
    target[key] = int(round(value)) if as_int else value


def clamp_probability(root: dict[str, Any], path: tuple[str, ...]) -> None:
    clamp_number(root, path, 0.0, 1.0)


def candidate_run_label(candidate: dict[str, Any], fallback: str) -> str:
    openpanel = candidate.get("openpanel")
    openpanel = openpanel if isinstance(openpanel, dict) else {}
    raw = (
        candidate.get("exp_name")
        or openpanel.get("name")
        or openpanel.get("experiment")
        or fallback
    )
    label = str(raw).strip()
    label = re.sub(r"^(asi[-_]?stage2[-_]?)", "", label, flags=re.IGNORECASE)
    label = re.sub(r"^(stage2[-_]?)", "", label, flags=re.IGNORECASE)
    label = re.sub(r"^(asi[-_]?)", "", label, flags=re.IGNORECASE)
    label = re.sub(r"[^a-zA-Z0-9_.-]+", "_", label).strip("._-")
    return (label or fallback)[:96]


def normalize_config(candidate: dict[str, Any], step_dir: Path) -> dict[str, Any]:
    base = yaml.safe_load(BASE_CONFIG.read_text(encoding="utf-8")) or {}
    config = deep_merge(base, candidate)

    step_name = step_dir.name
    safe_step = re.sub(r"[^a-zA-Z0-9_.-]+", "_", step_name)
    run_label = candidate_run_label(candidate, safe_step)
    safe_label = re.sub(r"[^a-zA-Z0-9_.-]+", "_", run_label)
    timestamp = int(time.time())
    exp_name = f"asi_{safe_label}_{safe_step}_{timestamp}"

    config["exp_name"] = exp_name
    config["output_dir"] = str(OUTPUT_ROOT)
    config["checkpoint_dir"] = None
    config["resume_from"] = None
    config["start_stage"] = 2
    config["end_stage"] = 2
    config["stage1_checkpoint"] = str(BASE_STAGE1_CHECKPOINT)
    config["device"] = "cuda"
    config["distributed"] = False
    config["world_size"] = 1
    config["local_rank"] = 0
    config["compute_fid"] = False
    config["num_val_samples"] = int(config.get("num_val_samples", 256))
    config["num_val_samples"] = max(64, min(512, config["num_val_samples"]))

    config["model"] = copy.deepcopy(FIXED_MODEL)

    data = config.setdefault("data", {})
    data["train_data_dir"] = "/home/cyprian/Work/datasets/clean_64"
    data["val_data_dir"] = "/home/cyprian/Work/datasets/clean_64"
    data["resolution_buckets"] = [64]
    data["ensure_divisible_by"] = 16
    data["longest_side_resize"] = True
    data["num_workers"] = max(0, min(8, int(data.get("num_workers", 4))))

    for key in [
        "horizontal_flip_prob",
        "vertical_flip_prob",
        "color_jitter_prob",
        "random_crop_prob",
        "blur_prob",
        "jpeg_prob",
    ]:
        clamp_probability(config, ("data", key))
    for key in ["brightness_jitter", "contrast_jitter", "saturation_jitter"]:
        clamp_number(config, ("data", key), 0.0, 0.2)
    clamp_number(config, ("data", "hue_jitter"), 0.0, 0.05)
    clamp_number(config, ("data", "random_crop_scale_min"), 0.7, 1.0)
    clamp_number(config, ("data", "random_crop_scale_max"), 0.7, 1.0)
    if data.get("random_crop_scale_max", 1.0) < data.get("random_crop_scale_min", 0.85):
        data["random_crop_scale_max"] = data["random_crop_scale_min"]
    clamp_number(config, ("data", "gaussian_noise_std"), 0.0, 0.03)
    clamp_number(config, ("data", "blur_kernel_size"), 3, 9, as_int=True)
    clamp_number(config, ("data", "jpeg_quality_min"), 60, 100, as_int=True)
    clamp_number(config, ("data", "jpeg_quality_max"), 60, 100, as_int=True)
    if data.get("jpeg_quality_max", 100) < data.get("jpeg_quality_min", 85):
        data["jpeg_quality_max"] = data["jpeg_quality_min"]

    stage2 = config.setdefault("stage2", {})
    clamp_number(config, ("stage2", "max_iterations"), 500, 3000, as_int=True)
    clamp_number(config, ("stage2", "batch_size"), 4, 64, as_int=True)
    clamp_number(config, ("stage2", "gradient_accumulation_steps"), 1, 8, as_int=True)
    clamp_number(config, ("stage2", "learning_rate"), 1e-6, 5e-4)
    clamp_number(config, ("stage2", "disc_learning_rate"), 1e-6, 5e-4)
    clamp_number(config, ("stage2", "weight_decay"), 0.0, 0.1)
    clamp_number(config, ("stage2", "warmup_iterations"), 0, 1000, as_int=True)
    clamp_number(config, ("stage2", "min_lr"), 0.0, 1e-5)
    clamp_number(config, ("stage2", "ema_decay"), 0.9, 0.99999)
    clamp_number(config, ("stage2", "validate_every"), 100, 1000, as_int=True)
    clamp_number(config, ("stage2", "checkpoint_every"), 500, 3000, as_int=True)
    clamp_number(config, ("stage2", "log_every"), 10, 200, as_int=True)

    max_iterations = int(stage2.get("max_iterations", 1000))
    stage2["validate_every"] = min(
        int(stage2.get("validate_every", 500)),
        max(100, max_iterations // 2),
    )
    stage2["checkpoint_every"] = max_iterations

    for key in [
        "lambda_l1",
        "lambda_mse",
        "lambda_charbonnier",
        "lambda_huber",
        "lambda_gradient",
        "lambda_laplacian",
        "lambda_ssim",
        "lambda_perceptual",
        "lambda_flow",
        "lambda_gram",
    ]:
        clamp_number(config, ("stage2", key), 0.0, 5.0)
    clamp_number(config, ("stage2", "lambda_adversarial"), 0.0, 0.2)
    clamp_number(config, ("stage2", "charbonnier_eps"), 1e-6, 0.1)
    clamp_number(config, ("stage2", "huber_delta"), 1e-4, 0.5)
    clamp_number(config, ("stage2", "disc_start_iteration"), 0, max_iterations, as_int=True)
    stage2["adversarial_loss_type"] = stage2.get("adversarial_loss_type", "hinge")
    if stage2["adversarial_loss_type"] not in {"hinge", "non-saturating", "least-squares"}:
        stage2["adversarial_loss_type"] = "hinge"
    if float(stage2.get("lambda_adversarial", 0.0)) <= 0:
        stage2["disc_start_iteration"] = 1_000_000

    openpanel = config.setdefault("openpanel", {})
    openpanel["enabled"] = True
    openpanel["project"] = OPENPANEL_PROJECT
    openpanel["experiment"] = run_label
    openpanel["name"] = run_label
    openpanel["tags"] = sorted(set(
        list(openpanel.get("tags") or [])
        + ["asi", "asi-evolve", "psnr-target-30", "clean_64", step_name, run_label]
    ))
    openpanel["silent"] = True
    openpanel["log_data"] = True

    return config


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def fail(output_path: Path, message: str, *, stdout: str = "", stderr: str = "") -> int:
    write_json(output_path, {
        "success": False,
        "eval_score": 0.0,
        "score": 0.0,
        "target_reached": False,
        "error": message,
        "temp": {
            "stdout": stdout[-20000:],
            "stderr": stderr[-20000:],
        },
    })
    return 0


def run_training(config_path: Path, step_dir: Path, timeout: int | None) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    cmd = [
        sys.executable,
        str(TRAINING_ROOT / "train.py"),
        "--config",
        str(config_path),
    ]
    completed = subprocess.run(
        cmd,
        cwd=str(COSMOS_ROOT),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
    )
    (step_dir / "train.stdout").write_text(completed.stdout, encoding="utf-8")
    (step_dir / "train.stderr").write_text(completed.stderr, encoding="utf-8")
    return completed


def evaluate_best_checkpoint(config_path: Path) -> tuple[dict[str, Any], Path]:
    sys.path.insert(0, str(COSMOS_ROOT))

    import torch
    from cosmos_tokenizer.networks.continuous_image import ContinuousImageTokenizer
    from training.config import load_config
    from training.dataset import create_dataloader
    from training.evaluate import evaluate_reconstruction
    from training.losses import CombinedLoss
    from training.train import evaluate_loss_components
    from training.utils import ExponentialMovingAverage

    cfg = load_config(config_path)
    checkpoint_dir = Path(cfg.checkpoint_dir)
    best_checkpoint = checkpoint_dir / "stage2_checkpoint_best.pt"
    if not best_checkpoint.exists():
        candidates = sorted(glob.glob(str(checkpoint_dir / "stage2_checkpoint_*.pt")))
        if not candidates:
            raise FileNotFoundError(f"no Stage 2 checkpoints found in {checkpoint_dir}")
        best_checkpoint = Path(candidates[-1])

    device = torch.device(cfg.device)
    model = ContinuousImageTokenizer(**vars(cfg.model)).to(device)
    checkpoint = torch.load(best_checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    used_ema_weights = False
    if checkpoint.get("ema_state_dict") is not None:
        ema = ExponentialMovingAverage(model, decay=cfg.stage2.ema_decay)
        ema.load_state_dict(checkpoint["ema_state_dict"])
        ema.apply_shadow()
        used_ema_weights = True

    val_loader = create_dataloader(
        data_dir=cfg.data.val_data_dir,
        batch_size=cfg.stage2.batch_size,
        resolution_buckets=cfg.data.resolution_buckets,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        persistent_workers=False,
        shuffle=False,
        use_augmentation=False,
        ensure_divisible_by=cfg.data.ensure_divisible_by,
        horizontal_flip_prob=0.0,
        vertical_flip_prob=0.0,
        color_jitter=False,
        color_jitter_prob=0.0,
        random_crop_prob=0.0,
        gaussian_noise_std=0.0,
        blur_prob=0.0,
        jpeg_prob=0.0,
        longest_side_resize=cfg.data.longest_side_resize,
    )

    criterion = CombinedLoss(
        stage=2,
        lambda_l1=cfg.stage2.lambda_l1,
        lambda_mse=cfg.stage2.lambda_mse,
        lambda_charbonnier=cfg.stage2.lambda_charbonnier,
        lambda_huber=cfg.stage2.lambda_huber,
        lambda_gradient=cfg.stage2.lambda_gradient,
        lambda_laplacian=cfg.stage2.lambda_laplacian,
        lambda_ssim=cfg.stage2.lambda_ssim,
        lambda_perceptual=cfg.stage2.lambda_perceptual,
        lambda_flow=cfg.stage2.lambda_flow,
        lambda_gram=cfg.stage2.lambda_gram,
        lambda_adversarial=cfg.stage2.lambda_adversarial,
        charbonnier_eps=cfg.stage2.charbonnier_eps,
        huber_delta=cfg.stage2.huber_delta,
        adversarial_loss_type=cfg.stage2.adversarial_loss_type,
        perceptual_layers=cfg.stage1.perceptual_layers,
        perceptual_weights=cfg.stage1.perceptual_weights,
    ).to(device)

    amp_dtype = getattr(torch, cfg.stage2.dtype) if cfg.stage2.dtype != "float32" else torch.float32
    use_amp = cfg.stage2.use_amp and cfg.stage2.dtype != "float32"
    metrics = evaluate_reconstruction(
        model=model,
        dataloader=val_loader,
        device=device,
        num_samples=cfg.num_val_samples,
        compute_fid_score=False,
    )
    loss_metrics = evaluate_loss_components(
        model=model,
        dataloader=val_loader,
        criterion=criterion,
        device=device,
        num_samples=cfg.num_val_samples,
        amp_dtype=amp_dtype,
        use_amp=use_amp,
    )
    metrics.update(loss_metrics)
    metrics["used_ema_weights"] = float(used_ema_weights)
    return metrics, best_checkpoint


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate_path", type=Path)
    parser.add_argument("output_path", type=Path)
    parser.add_argument("--timeout", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    step_dir = args.output_path.parent
    step_dir.mkdir(parents=True, exist_ok=True)

    try:
        candidate = load_yaml_text(args.candidate_path)
        config = normalize_config(candidate, step_dir)
        normalized_config = step_dir / "candidate_config.yaml"
        normalized_config.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    except Exception as exc:
        return fail(args.output_path, f"config normalization failed: {exc}")

    if args.dry_run:
        write_json(args.output_path, {
            "success": True,
            "eval_score": 0.0,
            "score": 0.0,
            "target_reached": False,
            "dry_run": True,
            "candidate_config_path": str(normalized_config),
            "openpanel_enabled": config["openpanel"]["enabled"],
            "openpanel_project": config["openpanel"]["project"],
            "openpanel_experiment": config["openpanel"]["experiment"],
            "openpanel_name": config["openpanel"].get("name"),
        })
        return 0

    if not BASE_STAGE1_CHECKPOINT.exists():
        return fail(args.output_path, f"missing Stage 1 checkpoint: {BASE_STAGE1_CHECKPOINT}")

    try:
        completed = run_training(normalized_config, step_dir, args.timeout)
    except subprocess.TimeoutExpired as exc:
        return fail(
            args.output_path,
            f"training timed out after {args.timeout}s",
            stdout=exc.stdout or "",
            stderr=exc.stderr or "",
        )

    if completed.returncode != 0:
        return fail(
            args.output_path,
            f"training failed with exit code {completed.returncode}",
            stdout=completed.stdout,
            stderr=completed.stderr,
        )

    try:
        metrics, best_checkpoint = evaluate_best_checkpoint(normalized_config)
    except Exception as exc:
        return fail(args.output_path, f"post-training evaluation failed: {exc}")

    psnr = float(metrics.get("psnr", 0.0))
    val_l1 = float(metrics.get("loss_l1", 1.0))
    eval_score = psnr - 10.0 * val_l1
    target_reached = psnr >= PSNR_TARGET

    result = {
        "success": True,
        "eval_score": eval_score,
        "score": eval_score,
        "target_reached": target_reached,
        "target_psnr": PSNR_TARGET,
        "val_psnr": psnr,
        "val_l1": val_l1,
        "val_ssim": float(metrics.get("ssim", 0.0)),
        "metrics": {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))},
        "best_checkpoint": str(best_checkpoint),
        "candidate_config_path": str(normalized_config),
        "openpanel_enabled": config["openpanel"]["enabled"],
        "openpanel_project": config["openpanel"]["project"],
        "openpanel_experiment": config["openpanel"]["experiment"],
        "openpanel_name": config["openpanel"].get("name"),
        "openpanel_tags": config["openpanel"]["tags"],
    }
    write_json(args.output_path, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
