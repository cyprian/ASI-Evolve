# Cosmos Tokenizer CI16 PSNR ASI Experiment

This ASI-Evolve experiment searches Stage 2 fine-tuning YAML configs for the
Cosmos CI16 tokenizer. It keeps the Baseline_64 architecture fixed and starts
each candidate from:

```text
/home/cyprian/Work/projects/cosmos-tokenizer/training/pretrained/Baseline_64/checkpoints/stage1_checkpoint_best.pt
```

Every candidate is normalized by `evaluator.py` before training:

- fixed architecture and data paths
- `start_stage: 2`, `end_stage: 2`
- scouting Stage 2 evaluation budget, clamped to 500-3000 iterations
- OpenPanel ML forced on:
  - project: `cosmos-tokenizer`
  - tags include `asi`, `asi-evolve`, `psnr-target-30`, `clean_64`

Run from the ASI-Evolve repo root:

```bash
python main.py \
  --experiment cosmos_tokenizer_ci16_psnr \
  --steps 20 \
  --sample-n 3 \
  --eval-script "$PWD/experiments/cosmos_tokenizer_ci16_psnr/eval.sh"
```

After ASI scouts promising configs, promote the top 3-5 YAMLs manually with `stage2.max_iterations: 10000`, `validate_every: 1000`, and `checkpoint_every: 10000` for a more reliable comparison.

The evaluator reports:

```text
eval_score = validation_psnr - 10.0 * validation_loss_l1
```

and records whether `validation_psnr >= 30.0`.
