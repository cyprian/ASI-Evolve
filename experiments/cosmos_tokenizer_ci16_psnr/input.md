# Cosmos CI16 Tokenizer PSNR Research

## Objective

Scout Stage 2 fine-tuning configurations for the 16x compressed continuous-image tokenizer that improve validation reconstruction quality on `clean_64`.

Target:
- Reach validation PSNR >= 32.0 dB.
- Also reduce validation L1 where possible.

Primary score used by ASI-Evolve:

```text
eval_score = validation_psnr - 10.0 * validation_loss_l1
```

## Fixed Architecture And Checkpoint

Keep the model architecture compatible with the existing Stage 1 checkpoint:

```yaml
model:
  spatial_compression: 16
  latent_channels: 16
  z_channels: 16
  z_factor: 1
  patch_size: 4
  patch_method: haar
  channels: 64
  channels_mult: [1, 2, 2]
  num_res_blocks: 1
  attn_resolutions: [16]
  formulation: AE
  in_channels: 3
  out_channels: 3
  resolution: 64
```

The evaluator always starts Stage 2 from:

```text
/home/cyprian/Work/projects/cosmos-tokenizer/training/pretrained/Baseline_64/checkpoints/stage1_checkpoint_best.pt
```

## Allowed Search Surface

Change Stage 2 optimization and loss settings:

```yaml
stage2:
  max_iterations
  batch_size
  gradient_accumulation_steps
  learning_rate
  weight_decay
  betas
  disc_learning_rate
  warmup_iterations
  lr_schedule
  min_lr
  lambda_l1
  lambda_mse
  lambda_charbonnier
  lambda_huber
  lambda_gradient
  lambda_laplacian
  lambda_ssim
  lambda_perceptual
  charbonnier_eps
  huber_delta
  lambda_flow
  lambda_gram
  lambda_adversarial
  adversarial_loss_type
  disc_start_iteration
  ema_decay
```

Change mild training augmentations:

```yaml
data:
  horizontal_flip_prob
  vertical_flip_prob
  color_jitter
  color_jitter_prob
  brightness_jitter
  contrast_jitter
  saturation_jitter
  hue_jitter
  random_crop_prob
  random_crop_scale_min
  random_crop_scale_max
  gaussian_noise_std
  blur_prob
  blur_kernel_size
  jpeg_prob
  jpeg_quality_min
  jpeg_quality_max
```

## Search Phases

This ASI run is a scouting phase. Candidates should usually train for 1000-3000 Stage 2 iterations. The goal is to find promising loss, optimization, and augmentation families, not to prove the final PSNR ceiling. Promote the best 3-5 configs later with 10000 Stage 2 iterations for confirmation.

## Important Constraints

- Output a complete YAML configuration only.
- Do not change the model architecture.
- Do not change data paths.
- Do not disable OpenPanel reporting. The evaluator forces OpenPanel on, but the YAML should also keep it enabled.
- Prefer PSNR-aligned losses first: MSE, L1, Charbonnier/Huber, low edge loss.
- Use adversarial and Gram losses cautiously; they can improve visual sharpness while hurting PSNR.
- Keep augmentations mild because the validation target is clean reconstruction.
