
## Sweep started 2026-10-14 09:47

Start: round-1 settings (resnet18_nobw), val mAP 0.236. Keep a change only if it adds > 0.01.

### Stage 1: input resolution (current: `round-1 settings`, val mAP 0.236)
- r2_s1_res192: `--size 192` -> val mAP 0.267
- r2_s1_up192: `--input-size 192` -> val mAP 0.275
- **Kept r2_s1_up192** (+0.039 over round 1 so far).

### Stage 2: learning rates (current: `--input-size 192`, val mAP 0.275)
- r2_s2_blr0.1: `--input-size 192 --backbone-lr-mult 0.1` -> val mAP 0.271
- r2_s2_lr1e-4: `--input-size 192 --lr 1e-4` -> val mAP 0.262
- No variant beat r2_s1_up192 by > 0.01; keeping current settings.

### Stage 3: regularization (current: `--input-size 192`, val mAP 0.275)

## Sweep started 2026-10-14 11:29

Start: round-1 settings (resnet18_nobw), val mAP 0.236. Keep a change only if it adds > 0.01.

### Stage 1: input resolution (current: `round-1 settings`, val mAP 0.236)
- r2_s1_res192: `--size 192` -> val mAP 0.267
- r2_s1_up192: `--input-size 192` -> val mAP 0.275
- **Kept r2_s1_up192** (+0.039 over round 1 so far).

### Stage 2: learning rates (current: `--input-size 192`, val mAP 0.275)
- r2_s2_blr0.1: `--input-size 192 --backbone-lr-mult 0.1` -> val mAP 0.271
- r2_s2_lr1e-4: `--input-size 192 --lr 1e-4` -> val mAP 0.262
- No variant beat r2_s1_up192 by > 0.01; keeping current settings.

### Stage 3: regularization (current: `--input-size 192`, val mAP 0.275)
- r2_s3_mixup: `--input-size 192 --mixup 0.4` -> val mAP 0.279
- r2_s3_reg: `--input-size 192 --dropout 0.5 --weight-decay 0.05` -> val mAP 0.279
- r2_s3_mixup_reg: `--input-size 192 --mixup 0.4 --dropout 0.5 --weight-decay 0.05` -> val mAP 0.289
- **Kept r2_s3_mixup_reg** (+0.053 over round 1 so far).

### Stage 4: confirmation with seeds (final settings: `--input-size 192 --mixup 0.4 --dropout 0.5 --weight-decay 0.05`)
- final (no BW): val mAP 0.281 +/- 0.008 over seeds (0.290, 0.274, 0.279)
- round-1 settings (no BW): val mAP 0.232 +/- 0.008 over seeds (0.236, 0.238, 0.223)
- final with BW (ablation): val mAP 0.284 +/- 0.009 over seeds (0.289, 0.290, 0.274)

Sweep finished 2026-10-14 12:56. Final settings: `--input-size 192 --mixup 0.4 --dropout 0.5 --weight-decay 0.05`.
