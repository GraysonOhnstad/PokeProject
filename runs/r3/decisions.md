
## Round 3 sweep started 2026-10-14 21:49 (backbones: resnet18, convnext_tiny, resnet50; 20 Optuna trials each; 5-fold CV; no pruning)

### resnet18

- Stage 0 baseline (round-2 settings): mAP 0.282 (folds +/- 0.019), F1 0.335, exact 0.104, top type 0.381, predicts dual 67%, train-val gap 0.591 (11.2M params)
  - Stage 1 feature tap, tap_stage3: mAP 0.266 (folds +/- 0.024), F1 0.325, exact 0.088, top type 0.370, predicts dual 70%, train-val gap 0.352 (2.8M params)
  - No variant cleared the rule; keeping current settings.
  - Stage 2 attention pooling, attn_fixed: mAP 0.283 (folds +/- 0.026), F1 0.337, exact 0.084, top type 0.386, predicts dual 81%, train-val gap 0.581 (11.2M params)
  - Stage 2 attention pooling, attn_learned: mAP 0.289 (folds +/- 0.019), F1 0.336, exact 0.093, top type 0.380, predicts dual 80%, train-val gap 0.602 (11.2M params)
  - No variant cleared the rule; keeping current settings.
  - Stage 3 auxiliary heads, aux0.1: mAP 0.292 (folds +/- 0.029), F1 0.346, exact 0.081, top type 0.398, predicts dual 84%, train-val gap 0.588 (11.2M params)
  - Stage 3 auxiliary heads, aux0.5: mAP 0.305 (folds +/- 0.019), F1 0.365, exact 0.094, top type 0.417, predicts dual 83%, train-val gap 0.467 (11.2M params)
  - **Kept aux0.5.**
  - Stage 4 set-valued output, set: mAP 0.287 (folds +/- 0.021), F1 0.353, exact 0.098, top type 0.390, predicts dual 85%, train-val gap 0.710 (11.2M params)
  - Stage 4 set-valued output, set_boost: mAP 0.282 (folds +/- 0.021), F1 0.351, exact 0.122, top type 0.390, predicts dual 64%, train-val gap 0.712 (11.2M params)
  - No variant cleared the rule; keeping current settings.

  Stage 5 Optuna: 20 trials, structure {'tap': 'final', 'pool': 'avg', 'learn_lambda': False, 'head': 'sigmoid', 'pair_boost': False}
  - Optuna trial 0: mAP 0.305 (folds +/- 0.019), F1 0.365, exact 0.094, top type 0.417, predicts dual 83%, train-val gap 0.467 (11.2M params)  `{"lr_scale": 1.0, "backbone_lr_mult": 1.0, "weight_decay": 0.05, "dropout": 0.5, "mixup": 0.4, "epochs": 60, "bg_prob": 0.0, "aug_strong": false, "weight_alpha": 1.0, "aux_weight": 0.5}`
  - Optuna trial 1: mAP 0.310 (folds +/- 0.023), F1 0.368, exact 0.122, top type 0.416, predicts dual 70%, train-val gap 0.458 (11.2M params)  `{"lr_scale": 0.7572, "backbone_lr_mult": 0.426, "weight_decay": 0.0311, "dropout": 0.3814, "mixup": 0.4237, "epochs": 44, "bg_prob": 0.3501, "aug_strong": true, "weight_alpha": 0.3834, "aux_weight": 0.4818}`
  - Optuna trial 2: mAP 0.307 (folds +/- 0.023), F1 0.360, exact 0.113, top type 0.411, predicts dual 72%, train-val gap 0.500 (11.2M params)  `{"lr_scale": 0.7036, "backbone_lr_mult": 0.2742, "weight_decay": 0.1963, "dropout": 0.0497, "mixup": 0.0871, "epochs": 15, "bg_prob": 0.6661, "aug_strong": true, "weight_alpha": 0.9786, "aux_weight": 0.4945}`
  - Optuna trial 3: mAP 0.307 (folds +/- 0.021), F1 0.371, exact 0.111, top type 0.421, predicts dual 75%, train-val gap 0.503 (11.2M params)  `{"lr_scale": 0.5487, "backbone_lr_mult": 0.5182, "weight_decay": 0.002, "dropout": 0.4479, "mixup": 0.1434, "epochs": 58, "bg_prob": 0.4175, "aug_strong": false, "weight_alpha": 0.7742, "aux_weight": 0.1485}`
  - Optuna trial 4: mAP 0.295 (folds +/- 0.015), F1 0.353, exact 0.077, top type 0.390, predicts dual 94%, train-val gap 0.479 (11.2M params)  `{"lr_scale": 0.8141, "backbone_lr_mult": 0.0529, "weight_decay": 0.0339, "dropout": 0.4285, "mixup": 0.6169, "epochs": 58, "bg_prob": 0.5455, "aug_strong": true, "weight_alpha": 0.6976, "aux_weight": 0.0371}`
  - Optuna trial 5: mAP 0.294 (folds +/- 0.013), F1 0.357, exact 0.106, top type 0.396, predicts dual 77%, train-val gap 0.550 (11.2M params)  `{"lr_scale": 1.17, "backbone_lr_mult": 0.3728, "weight_decay": 0.0033, "dropout": 0.0902, "mixup": 0.3154, "epochs": 31, "bg_prob": 0.4562, "aug_strong": true, "weight_alpha": 0.102, "aux_weight": 0.0624}`
  - Optuna trial 6: mAP 0.309 (folds +/- 0.025), F1 0.369, exact 0.104, top type 0.419, predicts dual 78%, train-val gap 0.587 (11.2M params)  `{"lr_scale": 0.1813, "backbone_lr_mult": 0.3537, "weight_decay": 0.0042, "dropout": 0.3264, "mixup": 0.2444, "epochs": 22, "bg_prob": 0.0883, "aug_strong": false, "weight_alpha": 0.1966, "aux_weight": 0.1093}`
  - Optuna trial 7: mAP 0.289 (folds +/- 0.015), F1 0.346, exact 0.072, top type 0.386, predicts dual 94%, train-val gap 0.553 (11.2M params)  `{"lr_scale": 2.0667, "backbone_lr_mult": 0.0669, "weight_decay": 0.119, "dropout": 0.0673, "mixup": 0.9765, "epochs": 36, "bg_prob": 0.7814, "aug_strong": true, "weight_alpha": 0.0392, "aux_weight": 0.0809}`
  - Optuna trial 8: mAP 0.312 (folds +/- 0.023), F1 0.375, exact 0.110, top type 0.418, predicts dual 82%, train-val gap 0.500 (11.2M params)  `{"lr_scale": 0.8627, "backbone_lr_mult": 0.2738, "weight_decay": 0.1689, "dropout": 0.4036, "mixup": 0.6391, "epochs": 35, "bg_prob": 0.3465, "aug_strong": true, "weight_alpha": 0.2477, "aux_weight": 0.3417}`
  - Optuna trial 9: mAP 0.289 (folds +/- 0.013), F1 0.348, exact 0.085, top type 0.397, predicts dual 85%, train-val gap 0.637 (11.2M params)  `{"lr_scale": 3.5209, "backbone_lr_mult": 0.1983, "weight_decay": 0.0302, "dropout": 0.3049, "mixup": 0.6562, "epochs": 42, "bg_prob": 0.6046, "aug_strong": true, "weight_alpha": 0.4096, "aux_weight": 0.098}`
  - Optuna trial 10: mAP 0.298 (folds +/- 0.018), F1 0.364, exact 0.106, top type 0.407, predicts dual 81%, train-val gap 0.487 (11.2M params)  `{"lr_scale": 0.7814, "backbone_lr_mult": 0.1423, "weight_decay": 0.1016, "dropout": 0.195, "mixup": 0.6753, "epochs": 43, "bg_prob": 0.5902, "aug_strong": true, "weight_alpha": 0.3246, "aux_weight": 0.6845}`
  - Optuna trial 11: mAP 0.305 (folds +/- 0.016), F1 0.364, exact 0.108, top type 0.415, predicts dual 75%, train-val gap 0.485 (11.2M params)  `{"lr_scale": 1.4068, "backbone_lr_mult": 0.1893, "weight_decay": 0.1087, "dropout": 0.3898, "mixup": 0.7584, "epochs": 42, "bg_prob": 0.307, "aug_strong": true, "weight_alpha": 0.3467, "aux_weight": 0.6957}`
  - Optuna trial 12: mAP 0.307 (folds +/- 0.014), F1 0.375, exact 0.104, top type 0.422, predicts dual 77%, train-val gap 0.578 (11.2M params)  `{"lr_scale": 0.536, "backbone_lr_mult": 0.5532, "weight_decay": 0.1858, "dropout": 0.3206, "mixup": 0.6194, "epochs": 51, "bg_prob": 0.1729, "aug_strong": true, "weight_alpha": 0.508, "aux_weight": 0.1804}`
  - Optuna trial 13: mAP 0.305 (folds +/- 0.021), F1 0.356, exact 0.103, top type 0.402, predicts dual 77%, train-val gap 0.495 (11.2M params)  `{"lr_scale": 1.1703, "backbone_lr_mult": 0.3301, "weight_decay": 0.0789, "dropout": 0.4833, "mixup": 0.5663, "epochs": 26, "bg_prob": 0.446, "aug_strong": true, "weight_alpha": 0.1981, "aux_weight": 0.5278}`
  - Optuna trial 14: mAP 0.309 (folds +/- 0.018), F1 0.372, exact 0.099, top type 0.416, predicts dual 81%, train-val gap 0.575 (11.2M params)  `{"lr_scale": 0.4809, "backbone_lr_mult": 0.7225, "weight_decay": 0.0332, "dropout": 0.2547, "mixup": 0.6749, "epochs": 38, "bg_prob": 0.0565, "aug_strong": true, "weight_alpha": 0.2325, "aux_weight": 0.3593}`
  - Optuna trial 15: mAP 0.311 (folds +/- 0.015), F1 0.372, exact 0.113, top type 0.411, predicts dual 78%, train-val gap 0.524 (11.2M params)  `{"lr_scale": 0.8763, "backbone_lr_mult": 0.6834, "weight_decay": 0.0094, "dropout": 0.3996, "mixup": 0.3044, "epochs": 43, "bg_prob": 0.4649, "aug_strong": true, "weight_alpha": 0.4117, "aux_weight": 0.3566}`
  - Optuna trial 16: mAP 0.306 (folds +/- 0.017), F1 0.372, exact 0.106, top type 0.416, predicts dual 82%, train-val gap 0.553 (11.2M params)  `{"lr_scale": 1.2076, "backbone_lr_mult": 0.2767, "weight_decay": 0.0148, "dropout": 0.3197, "mixup": 0.4302, "epochs": 45, "bg_prob": 0.3947, "aug_strong": true, "weight_alpha": 0.0541, "aux_weight": 0.3479}`
  - Optuna trial 17: mAP 0.301 (folds +/- 0.017), F1 0.363, exact 0.104, top type 0.408, predicts dual 82%, train-val gap 0.653 (11.2M params)  `{"lr_scale": 0.8265, "backbone_lr_mult": 0.8971, "weight_decay": 0.0473, "dropout": 0.4588, "mixup": 0.5201, "epochs": 41, "bg_prob": 0.5146, "aug_strong": true, "weight_alpha": 0.33, "aux_weight": 0.1303}`
  - Optuna trial 18: mAP 0.311 (folds +/- 0.025), F1 0.372, exact 0.107, top type 0.414, predicts dual 82%, train-val gap 0.555 (11.2M params)  `{"lr_scale": 0.5582, "backbone_lr_mult": 0.2415, "weight_decay": 0.0219, "dropout": 0.3224, "mixup": 0.5202, "epochs": 29, "bg_prob": 0.3228, "aug_strong": true, "weight_alpha": 0.2091, "aux_weight": 0.3032}`
  - Optuna trial 19: mAP 0.311 (folds +/- 0.019), F1 0.371, exact 0.100, top type 0.417, predicts dual 86%, train-val gap 0.493 (11.2M params)  `{"lr_scale": 0.3945, "backbone_lr_mult": 0.3928, "weight_decay": 0.0661, "dropout": 0.2673, "mixup": 0.5757, "epochs": 43, "bg_prob": 0.4136, "aug_strong": true, "weight_alpha": 0.1067, "aux_weight": 0.7201}`

  Stage 6: top 3 re-run with a second seed
  - trial 8: seed 0 mAP 0.312, seed 1 mAP 0.302 -> mean 0.307; F1 0.375 / 0.371
  - trial 15: seed 0 mAP 0.311, seed 1 mAP 0.304 -> mean 0.307; F1 0.372 / 0.359
  - trial 19: seed 0 mAP 0.311, seed 1 mAP 0.303 -> mean 0.307; F1 0.371 / 0.365
  - **resnet18 winner: trial 15, 10-model mean mAP 0.307.**
  (150 runs trained this session; remaining estimate ~21.2 h at 4.0 min/run)

### convnext_tiny

- Stage 0 baseline (round-2 settings): mAP 0.294 (folds +/- 0.024), F1 0.346, exact 0.107, top type 0.389, predicts dual 70%, train-val gap 0.638 (27.8M params)
  - Stage 1 feature tap, tap_stage3: mAP 0.287 (folds +/- 0.022), F1 0.343, exact 0.092, top type 0.380, predicts dual 73%, train-val gap 0.536 (12.4M params)
  - No variant cleared the rule; keeping current settings.
  - Stage 2 attention pooling, attn_fixed: mAP 0.300 (folds +/- 0.029), F1 0.359, exact 0.113, top type 0.398, predicts dual 71%, train-val gap 0.594 (27.8M params)
  - Stage 2 attention pooling, attn_learned: mAP 0.300 (folds +/- 0.028), F1 0.359, exact 0.111, top type 0.401, predicts dual 70%, train-val gap 0.597 (27.8M params)
  - No variant cleared the rule; keeping current settings.
  - Stage 3 auxiliary heads, aux0.1: mAP 0.308 (folds +/- 0.026), F1 0.360, exact 0.122, top type 0.407, predicts dual 62%, train-val gap 0.587 (27.9M params)
  - Stage 3 auxiliary heads, aux0.5: mAP 0.314 (folds +/- 0.021), F1 0.373, exact 0.126, top type 0.419, predicts dual 67%, train-val gap 0.445 (27.9M params)
  - **Kept aux0.5.**
  - Stage 4 set-valued output, set: mAP 0.325 (folds +/- 0.034), F1 0.386, exact 0.141, top type 0.424, predicts dual 70%, train-val gap 0.567 (27.9M params)
  - Stage 4 set-valued output, set_boost: mAP 0.324 (folds +/- 0.037), F1 0.384, exact 0.139, top type 0.423, predicts dual 70%, train-val gap 0.534 (27.9M params)
  - **Kept set.**

  Stage 5 Optuna: 20 trials, structure {'tap': 'final', 'pool': 'avg', 'learn_lambda': False, 'head': 'set', 'pair_boost': False}
  - Optuna trial 0: mAP 0.325 (folds +/- 0.034), F1 0.386, exact 0.141, top type 0.424, predicts dual 70%, train-val gap 0.567 (27.9M params)  `{"lr_scale": 1.0, "backbone_lr_mult": 1.0, "weight_decay": 0.05, "dropout": 0.5, "mixup": 0.4, "epochs": 60, "bg_prob": 0.0, "aug_strong": false, "weight_alpha": 1.0, "aux_weight": 0.5}`
  - Optuna trial 1: mAP 0.317 (folds +/- 0.026), F1 0.383, exact 0.137, top type 0.421, predicts dual 71%, train-val gap 0.584 (27.9M params)  `{"lr_scale": 0.7572, "backbone_lr_mult": 0.426, "weight_decay": 0.0311, "dropout": 0.3814, "mixup": 0.4237, "epochs": 44, "bg_prob": 0.3501, "aug_strong": true, "weight_alpha": 0.3834, "aux_weight": 0.4818}`
  - Optuna trial 2: mAP 0.314 (folds +/- 0.031), F1 0.377, exact 0.119, top type 0.423, predicts dual 77%, train-val gap 0.469 (27.9M params)  `{"lr_scale": 0.7036, "backbone_lr_mult": 0.2742, "weight_decay": 0.1963, "dropout": 0.0497, "mixup": 0.0871, "epochs": 15, "bg_prob": 0.6661, "aug_strong": true, "weight_alpha": 0.9786, "aux_weight": 0.4945}`
  - Optuna trial 3: mAP 0.311 (folds +/- 0.032), F1 0.370, exact 0.102, top type 0.412, predicts dual 85%, train-val gap 0.538 (27.9M params)  `{"lr_scale": 0.5487, "backbone_lr_mult": 0.5182, "weight_decay": 0.002, "dropout": 0.4479, "mixup": 0.1434, "epochs": 58, "bg_prob": 0.4175, "aug_strong": false, "weight_alpha": 0.7742, "aux_weight": 0.1485}`
  - Optuna trial 4: mAP 0.328 (folds +/- 0.027), F1 0.390, exact 0.133, top type 0.430, predicts dual 71%, train-val gap 0.500 (27.9M params)  `{"lr_scale": 0.8141, "backbone_lr_mult": 0.0529, "weight_decay": 0.0339, "dropout": 0.4285, "mixup": 0.6169, "epochs": 58, "bg_prob": 0.5455, "aug_strong": true, "weight_alpha": 0.6976, "aux_weight": 0.0371}`
  - Optuna trial 5: mAP 0.312 (folds +/- 0.026), F1 0.375, exact 0.121, top type 0.414, predicts dual 77%, train-val gap 0.591 (27.9M params)  `{"lr_scale": 1.17, "backbone_lr_mult": 0.3728, "weight_decay": 0.0033, "dropout": 0.0902, "mixup": 0.3154, "epochs": 31, "bg_prob": 0.4562, "aug_strong": true, "weight_alpha": 0.102, "aux_weight": 0.0624}`
  - Optuna trial 6: mAP 0.296 (folds +/- 0.038), F1 0.366, exact 0.108, top type 0.401, predicts dual 82%, train-val gap 0.391 (27.9M params)  `{"lr_scale": 0.1813, "backbone_lr_mult": 0.3537, "weight_decay": 0.0042, "dropout": 0.3264, "mixup": 0.2444, "epochs": 22, "bg_prob": 0.0883, "aug_strong": false, "weight_alpha": 0.1966, "aux_weight": 0.1093}`
  - Optuna trial 7: mAP 0.323 (folds +/- 0.033), F1 0.381, exact 0.137, top type 0.422, predicts dual 74%, train-val gap 0.545 (27.9M params)  `{"lr_scale": 2.0667, "backbone_lr_mult": 0.0669, "weight_decay": 0.119, "dropout": 0.0673, "mixup": 0.9765, "epochs": 36, "bg_prob": 0.7814, "aug_strong": true, "weight_alpha": 0.0392, "aux_weight": 0.0809}`
  - Optuna trial 8: mAP 0.332 (folds +/- 0.033), F1 0.384, exact 0.116, top type 0.427, predicts dual 83%, train-val gap 0.582 (27.9M params)  `{"lr_scale": 2.9751, "backbone_lr_mult": 0.1481, "weight_decay": 0.0196, "dropout": 0.2357, "mixup": 0.7075, "epochs": 45, "bg_prob": 0.489, "aug_strong": true, "weight_alpha": 0.9545, "aux_weight": 0.0648}`
  - Optuna trial 9: mAP 0.328 (folds +/- 0.034), F1 0.389, exact 0.130, top type 0.428, predicts dual 77%, train-val gap 0.592 (27.9M params)  `{"lr_scale": 3.6211, "backbone_lr_mult": 0.1664, "weight_decay": 0.0225, "dropout": 0.2598, "mixup": 0.6662, "epochs": 45, "bg_prob": 0.638, "aug_strong": true, "weight_alpha": 0.5155, "aux_weight": 0.073}`
  - Optuna trial 10: mAP 0.256 (folds +/- 0.021), F1 0.313, exact 0.072, top type 0.353, predicts dual 79%, train-val gap 0.147 (27.9M params)  `{"lr_scale": 0.1527, "backbone_lr_mult": 0.0995, "weight_decay": 0.0064, "dropout": 0.0406, "mixup": 0.9878, "epochs": 35, "bg_prob": 0.2308, "aug_strong": true, "weight_alpha": 0.6038, "aux_weight": 0.0383}`
  - Optuna trial 11: mAP 0.325 (folds +/- 0.031), F1 0.381, exact 0.110, top type 0.423, predicts dual 87%, train-val gap 0.556 (27.9M params)  `{"lr_scale": 3.4364, "backbone_lr_mult": 0.1273, "weight_decay": 0.0114, "dropout": 0.2807, "mixup": 0.6428, "epochs": 39, "bg_prob": 0.6364, "aug_strong": true, "weight_alpha": 0.2874, "aux_weight": 0.0891}`
  - Optuna trial 12: mAP 0.321 (folds +/- 0.025), F1 0.380, exact 0.113, top type 0.419, predicts dual 82%, train-val gap 0.496 (27.9M params)  `{"lr_scale": 0.8872, "backbone_lr_mult": 0.1596, "weight_decay": 0.0393, "dropout": 0.2788, "mixup": 0.5886, "epochs": 46, "bg_prob": 0.4224, "aug_strong": true, "weight_alpha": 0.9691, "aux_weight": 0.0388}`
  - Optuna trial 13: mAP 0.322 (folds +/- 0.034), F1 0.380, exact 0.119, top type 0.421, predicts dual 81%, train-val gap 0.587 (27.9M params)  `{"lr_scale": 2.059, "backbone_lr_mult": 0.2166, "weight_decay": 0.052, "dropout": 0.1757, "mixup": 0.9818, "epochs": 58, "bg_prob": 0.5502, "aug_strong": true, "weight_alpha": 0.9075, "aux_weight": 0.0605}`
  - Optuna trial 14: mAP 0.323 (folds +/- 0.032), F1 0.378, exact 0.138, top type 0.428, predicts dual 68%, train-val gap 0.560 (27.9M params)  `{"lr_scale": 3.1661, "backbone_lr_mult": 0.2666, "weight_decay": 0.0058, "dropout": 0.3909, "mixup": 0.7595, "epochs": 51, "bg_prob": 0.5697, "aug_strong": true, "weight_alpha": 0.7671, "aux_weight": 0.0362}`
  - Optuna trial 15: mAP 0.323 (folds +/- 0.031), F1 0.378, exact 0.125, top type 0.424, predicts dual 78%, train-val gap 0.587 (27.9M params)  `{"lr_scale": 3.0276, "backbone_lr_mult": 0.2259, "weight_decay": 0.0513, "dropout": 0.2228, "mixup": 0.6573, "epochs": 42, "bg_prob": 0.698, "aug_strong": true, "weight_alpha": 0.8169, "aux_weight": 0.0662}`
  - Optuna trial 16: mAP 0.323 (folds +/- 0.034), F1 0.382, exact 0.129, top type 0.427, predicts dual 76%, train-val gap 0.535 (27.9M params)  `{"lr_scale": 2.4713, "backbone_lr_mult": 0.1668, "weight_decay": 0.0094, "dropout": 0.1832, "mixup": 0.5416, "epochs": 53, "bg_prob": 0.5155, "aug_strong": true, "weight_alpha": 0.2391, "aux_weight": 0.0911}`
  - Optuna trial 17: mAP 0.327 (folds +/- 0.036), F1 0.378, exact 0.111, top type 0.424, predicts dual 87%, train-val gap 0.529 (27.9M params)  `{"lr_scale": 3.8986, "backbone_lr_mult": 0.1391, "weight_decay": 0.0645, "dropout": 0.3097, "mixup": 0.3434, "epochs": 36, "bg_prob": 0.5032, "aug_strong": true, "weight_alpha": 0.8333, "aux_weight": 0.0328}`

## Round 3 sweep started 2026-10-15 22:07 (backbones: resnet18, convnext_tiny, resnet50; 20 Optuna trials each; 5-fold CV; no pruning)

### resnet18

- Stage 0 baseline (round-2 settings): mAP 0.282 (folds +/- 0.019), F1 0.335, exact 0.104, top type 0.381, predicts dual 67%, train-val gap 0.591 (11.2M params)
  - Stage 1 feature tap, tap_stage3: mAP 0.266 (folds +/- 0.024), F1 0.325, exact 0.088, top type 0.370, predicts dual 70%, train-val gap 0.352 (2.8M params)
  - No variant cleared the rule; keeping current settings.
  - Stage 2 attention pooling, attn_fixed: mAP 0.283 (folds +/- 0.026), F1 0.337, exact 0.084, top type 0.386, predicts dual 81%, train-val gap 0.581 (11.2M params)
  - Stage 2 attention pooling, attn_learned: mAP 0.289 (folds +/- 0.019), F1 0.336, exact 0.093, top type 0.380, predicts dual 80%, train-val gap 0.602 (11.2M params)
  - No variant cleared the rule; keeping current settings.
  - Stage 3 auxiliary heads, aux0.1: mAP 0.292 (folds +/- 0.029), F1 0.346, exact 0.081, top type 0.398, predicts dual 84%, train-val gap 0.588 (11.2M params)
  - Stage 3 auxiliary heads, aux0.5: mAP 0.305 (folds +/- 0.019), F1 0.365, exact 0.094, top type 0.417, predicts dual 83%, train-val gap 0.467 (11.2M params)
  - **Kept aux0.5.**
  - Stage 4 set-valued output, set: mAP 0.287 (folds +/- 0.021), F1 0.353, exact 0.098, top type 0.390, predicts dual 85%, train-val gap 0.710 (11.2M params)
  - Stage 4 set-valued output, set_boost: mAP 0.282 (folds +/- 0.021), F1 0.351, exact 0.122, top type 0.390, predicts dual 64%, train-val gap 0.712 (11.2M params)
  - No variant cleared the rule; keeping current settings.

  Stage 6: top 3 re-run with a second seed
  - trial 8: seed 0 mAP 0.312, seed 1 mAP 0.302 -> mean 0.307; F1 0.375 / 0.371
  - trial 15: seed 0 mAP 0.311, seed 1 mAP 0.304 -> mean 0.307; F1 0.372 / 0.359
  - trial 19: seed 0 mAP 0.311, seed 1 mAP 0.303 -> mean 0.307; F1 0.371 / 0.365
  - **resnet18 winner: trial 15, 10-model mean mAP 0.307.**
  (0 runs trained this session; remaining estimate ?)

### convnext_tiny

- Stage 0 baseline (round-2 settings): mAP 0.294 (folds +/- 0.024), F1 0.346, exact 0.107, top type 0.389, predicts dual 70%, train-val gap 0.638 (27.8M params)
  - Stage 1 feature tap, tap_stage3: mAP 0.287 (folds +/- 0.022), F1 0.343, exact 0.092, top type 0.380, predicts dual 73%, train-val gap 0.536 (12.4M params)
  - No variant cleared the rule; keeping current settings.
  - Stage 2 attention pooling, attn_fixed: mAP 0.300 (folds +/- 0.029), F1 0.359, exact 0.113, top type 0.398, predicts dual 71%, train-val gap 0.594 (27.8M params)
  - Stage 2 attention pooling, attn_learned: mAP 0.300 (folds +/- 0.028), F1 0.359, exact 0.111, top type 0.401, predicts dual 70%, train-val gap 0.597 (27.8M params)
  - No variant cleared the rule; keeping current settings.
  - Stage 3 auxiliary heads, aux0.1: mAP 0.308 (folds +/- 0.026), F1 0.360, exact 0.122, top type 0.407, predicts dual 62%, train-val gap 0.587 (27.9M params)
  - Stage 3 auxiliary heads, aux0.5: mAP 0.314 (folds +/- 0.021), F1 0.373, exact 0.126, top type 0.419, predicts dual 67%, train-val gap 0.445 (27.9M params)
  - **Kept aux0.5.**
  - Stage 4 set-valued output, set: mAP 0.325 (folds +/- 0.034), F1 0.386, exact 0.141, top type 0.424, predicts dual 70%, train-val gap 0.567 (27.9M params)
  - Stage 4 set-valued output, set_boost: mAP 0.324 (folds +/- 0.037), F1 0.384, exact 0.139, top type 0.423, predicts dual 70%, train-val gap 0.534 (27.9M params)
  - **Kept set.**
  - Optuna trial 19: mAP 0.322 (folds +/- 0.026), F1 0.384, exact 0.110, top type 0.426, predicts dual 85%, train-val gap 0.560 (27.9M params)  `{"lr_scale": 1.3922, "backbone_lr_mult": 0.1486, "weight_decay": 0.0124, "dropout": 0.1856, "mixup": 0.6136, "epochs": 37, "bg_prob": 0.4448, "aug_strong": true, "weight_alpha": 0.5681, "aux_weight": 0.0798}`
  - Optuna trial 20: mAP 0.326 (folds +/- 0.031), F1 0.389, exact 0.130, top type 0.429, predicts dual 76%, train-val gap 0.619 (27.9M params)  `{"lr_scale": 2.1447, "backbone_lr_mult": 0.1644, "weight_decay": 0.1051, "dropout": 0.2827, "mixup": 0.7695, "epochs": 41, "bg_prob": 0.4509, "aug_strong": true, "weight_alpha": 0.4967, "aux_weight": 0.1083}`

  Stage 6: top 3 re-run with a second seed
  - trial 8: seed 0 mAP 0.332, seed 1 mAP 0.321 -> mean 0.326; F1 0.384 / 0.388
  - trial 9: seed 0 mAP 0.328, seed 1 mAP 0.319 -> mean 0.323; F1 0.389 / 0.378
  - trial 4: seed 0 mAP 0.328, seed 1 mAP 0.321 -> mean 0.325; F1 0.390 / 0.385
  - **convnext_tiny winner: trial 8, 10-model mean mAP 0.326.**
  (25 runs trained this session; remaining estimate ~16.5 h at 6.0 min/run)

### resnet50

- Stage 0 baseline (round-2 settings): mAP 0.305 (folds +/- 0.026), F1 0.359, exact 0.106, top type 0.409, predicts dual 76%, train-val gap 0.695 (23.5M params)
  - Stage 1 feature tap, tap_stage3: mAP 0.300 (folds +/- 0.026), F1 0.339, exact 0.073, top type 0.379, predicts dual 89%, train-val gap 0.472 (8.6M params)
  - No variant cleared the rule; keeping current settings.
  - Stage 2 attention pooling, attn_fixed: mAP 0.305 (folds +/- 0.033), F1 0.372, exact 0.103, top type 0.416, predicts dual 82%, train-val gap 0.695 (23.5M params)
  - Stage 2 attention pooling, attn_learned: mAP 0.296 (folds +/- 0.032), F1 0.362, exact 0.124, top type 0.403, predicts dual 71%, train-val gap 0.704 (23.5M params)
  - No variant cleared the rule; keeping current settings.
  - Stage 3 auxiliary heads, aux0.1: mAP 0.309 (folds +/- 0.027), F1 0.371, exact 0.127, top type 0.421, predicts dual 68%, train-val gap 0.691 (23.6M params)
  - Stage 3 auxiliary heads, aux0.5: mAP 0.316 (folds +/- 0.027), F1 0.379, exact 0.119, top type 0.428, predicts dual 73%, train-val gap 0.574 (23.6M params)
  - **Kept aux0.5.**
  - Stage 4 set-valued output, set: mAP 0.307 (folds +/- 0.026), F1 0.369, exact 0.123, top type 0.410, predicts dual 75%, train-val gap 0.692 (23.6M params)
  - Stage 4 set-valued output, set_boost: mAP 0.309 (folds +/- 0.035), F1 0.377, exact 0.123, top type 0.418, predicts dual 76%, train-val gap 0.656 (23.6M params)
  - No variant cleared the rule; keeping current settings.

  Stage 5 Optuna: 20 trials, structure {'tap': 'final', 'pool': 'avg', 'learn_lambda': False, 'head': 'sigmoid', 'pair_boost': False}
  - Optuna trial 0: mAP 0.316 (folds +/- 0.027), F1 0.379, exact 0.119, top type 0.428, predicts dual 73%, train-val gap 0.574 (23.6M params)  `{"lr_scale": 1.0, "backbone_lr_mult": 1.0, "weight_decay": 0.05, "dropout": 0.5, "mixup": 0.4, "epochs": 60, "bg_prob": 0.0, "aug_strong": false, "weight_alpha": 1.0, "aux_weight": 0.5}`
  - Optuna trial 1: mAP 0.317 (folds +/- 0.016), F1 0.379, exact 0.118, top type 0.430, predicts dual 73%, train-val gap 0.500 (23.6M params)  `{"lr_scale": 0.7572, "backbone_lr_mult": 0.426, "weight_decay": 0.0311, "dropout": 0.3814, "mixup": 0.4237, "epochs": 44, "bg_prob": 0.3501, "aug_strong": true, "weight_alpha": 0.3834, "aux_weight": 0.4818}`
  - Optuna trial 2: mAP 0.302 (folds +/- 0.020), F1 0.363, exact 0.120, top type 0.415, predicts dual 65%, train-val gap 0.484 (23.6M params)  `{"lr_scale": 0.7036, "backbone_lr_mult": 0.2742, "weight_decay": 0.1963, "dropout": 0.0497, "mixup": 0.0871, "epochs": 15, "bg_prob": 0.6661, "aug_strong": true, "weight_alpha": 0.9786, "aux_weight": 0.4945}`
  - Optuna trial 3: mAP 0.314 (folds +/- 0.024), F1 0.378, exact 0.113, top type 0.429, predicts dual 78%, train-val gap 0.604 (23.6M params)  `{"lr_scale": 0.5487, "backbone_lr_mult": 0.5182, "weight_decay": 0.002, "dropout": 0.4479, "mixup": 0.1434, "epochs": 58, "bg_prob": 0.4175, "aug_strong": false, "weight_alpha": 0.7742, "aux_weight": 0.1485}`
  - Optuna trial 4: mAP 0.288 (folds +/- 0.014), F1 0.353, exact 0.096, top type 0.402, predicts dual 78%, train-val gap 0.657 (23.6M params)  `{"lr_scale": 0.8141, "backbone_lr_mult": 0.0529, "weight_decay": 0.0339, "dropout": 0.4285, "mixup": 0.6169, "epochs": 58, "bg_prob": 0.5455, "aug_strong": true, "weight_alpha": 0.6976, "aux_weight": 0.0371}`
  - Optuna trial 5: mAP 0.307 (folds +/- 0.022), F1 0.372, exact 0.112, top type 0.415, predicts dual 76%, train-val gap 0.661 (23.6M params)  `{"lr_scale": 1.17, "backbone_lr_mult": 0.3728, "weight_decay": 0.0033, "dropout": 0.0902, "mixup": 0.3154, "epochs": 31, "bg_prob": 0.4562, "aug_strong": true, "weight_alpha": 0.102, "aux_weight": 0.0624}`
  - Optuna trial 6: mAP 0.306 (folds +/- 0.019), F1 0.373, exact 0.128, top type 0.419, predicts dual 70%, train-val gap 0.602 (23.6M params)  `{"lr_scale": 0.1813, "backbone_lr_mult": 0.3537, "weight_decay": 0.0042, "dropout": 0.3264, "mixup": 0.2444, "epochs": 22, "bg_prob": 0.0883, "aug_strong": false, "weight_alpha": 0.1966, "aux_weight": 0.1093}`
  - Optuna trial 7: mAP 0.296 (folds +/- 0.019), F1 0.368, exact 0.114, top type 0.415, predicts dual 76%, train-val gap 0.631 (23.6M params)  `{"lr_scale": 2.0667, "backbone_lr_mult": 0.0669, "weight_decay": 0.119, "dropout": 0.0673, "mixup": 0.9765, "epochs": 36, "bg_prob": 0.7814, "aug_strong": true, "weight_alpha": 0.0392, "aux_weight": 0.0809}`
  - Optuna trial 8: mAP 0.311 (folds +/- 0.017), F1 0.373, exact 0.109, top type 0.427, predicts dual 80%, train-val gap 0.521 (23.6M params)  `{"lr_scale": 0.8627, "backbone_lr_mult": 0.2738, "weight_decay": 0.1689, "dropout": 0.4036, "mixup": 0.6391, "epochs": 35, "bg_prob": 0.3465, "aug_strong": true, "weight_alpha": 0.2477, "aux_weight": 0.3417}`
  - Optuna trial 9: mAP 0.310 (folds +/- 0.020), F1 0.375, exact 0.122, top type 0.422, predicts dual 72%, train-val gap 0.600 (23.6M params)  `{"lr_scale": 0.3692, "backbone_lr_mult": 0.8774, "weight_decay": 0.0073, "dropout": 0.2534, "mixup": 0.2456, "epochs": 60, "bg_prob": 0.5048, "aug_strong": true, "weight_alpha": 0.4104, "aux_weight": 0.1494}`
  - Optuna trial 10: mAP 0.314 (folds +/- 0.023), F1 0.364, exact 0.104, top type 0.413, predicts dual 73%, train-val gap 0.531 (23.6M params)  `{"lr_scale": 1.7031, "backbone_lr_mult": 0.5953, "weight_decay": 0.0053, "dropout": 0.3188, "mixup": 0.1861, "epochs": 33, "bg_prob": 0.4964, "aug_strong": true, "weight_alpha": 0.3577, "aux_weight": 0.9366}`
  - Optuna trial 11: mAP 0.315 (folds +/- 0.015), F1 0.378, exact 0.114, top type 0.428, predicts dual 77%, train-val gap 0.576 (23.6M params)  `{"lr_scale": 0.6162, "backbone_lr_mult": 0.5565, "weight_decay": 0.0182, "dropout": 0.4084, "mixup": 0.5909, "epochs": 41, "bg_prob": 0.2675, "aug_strong": false, "weight_alpha": 0.5973, "aux_weight": 0.7781}`
  - Optuna trial 12: mAP 0.306 (folds +/- 0.016), F1 0.366, exact 0.092, top type 0.416, predicts dual 81%, train-val gap 0.430 (23.6M params)  `{"lr_scale": 1.8802, "backbone_lr_mult": 0.4506, "weight_decay": 0.1588, "dropout": 0.5373, "mixup": 0.7021, "epochs": 60, "bg_prob": 0.2228, "aug_strong": false, "weight_alpha": 0.8923, "aux_weight": 0.4953}`
  - Optuna trial 13: mAP 0.312 (folds +/- 0.025), F1 0.364, exact 0.091, top type 0.412, predicts dual 83%, train-val gap 0.572 (23.6M params)  `{"lr_scale": 0.674, "backbone_lr_mult": 0.9561, "weight_decay": 0.0766, "dropout": 0.2916, "mixup": 0.4752, "epochs": 54, "bg_prob": 0.057, "aug_strong": false, "weight_alpha": 0.9352, "aux_weight": 0.5209}`
  - Optuna trial 14: mAP 0.313 (folds +/- 0.032), F1 0.371, exact 0.131, top type 0.424, predicts dual 66%, train-val gap 0.644 (23.6M params)  `{"lr_scale": 2.0831, "backbone_lr_mult": 0.7802, "weight_decay": 0.0128, "dropout": 0.5903, "mixup": 0.5132, "epochs": 56, "bg_prob": 0.3486, "aug_strong": false, "weight_alpha": 0.7193, "aux_weight": 0.1297}`
  - Optuna trial 15: mAP 0.317 (folds +/- 0.023), F1 0.373, exact 0.105, top type 0.424, predicts dual 81%, train-val gap 0.628 (23.6M params)  `{"lr_scale": 1.1017, "backbone_lr_mult": 0.8445, "weight_decay": 0.0168, "dropout": 0.4933, "mixup": 0.286, "epochs": 50, "bg_prob": 0.3616, "aug_strong": false, "weight_alpha": 0.6488, "aux_weight": 0.3653}`
  - Optuna trial 16: mAP 0.315 (folds +/- 0.014), F1 0.362, exact 0.139, top type 0.418, predicts dual 57%, train-val gap 0.530 (23.6M params)  `{"lr_scale": 1.64, "backbone_lr_mult": 0.8846, "weight_decay": 0.0219, "dropout": 0.412, "mixup": 0.3279, "epochs": 41, "bg_prob": 0.3381, "aug_strong": false, "weight_alpha": 0.3807, "aux_weight": 0.9557}`
  - Optuna trial 17: mAP 0.316 (folds +/- 0.020), F1 0.362, exact 0.105, top type 0.409, predicts dual 77%, train-val gap 0.566 (23.6M params)  `{"lr_scale": 1.0663, "backbone_lr_mult": 0.6292, "weight_decay": 0.0389, "dropout": 0.5691, "mixup": 0.5242, "epochs": 49, "bg_prob": 0.3925, "aug_strong": false, "weight_alpha": 0.7327, "aux_weight": 0.2663}`
  - Optuna trial 18: mAP 0.319 (folds +/- 0.019), F1 0.379, exact 0.138, top type 0.429, predicts dual 62%, train-val gap 0.508 (23.6M params)  `{"lr_scale": 0.4989, "backbone_lr_mult": 0.3571, "weight_decay": 0.0184, "dropout": 0.3032, "mixup": 0.3754, "epochs": 37, "bg_prob": 0.3258, "aug_strong": true, "weight_alpha": 0.3009, "aux_weight": 0.3847}`
  - Optuna trial 19: mAP 0.312 (folds +/- 0.017), F1 0.370, exact 0.089, top type 0.416, predicts dual 91%, train-val gap 0.564 (23.6M params)  `{"lr_scale": 0.6249, "backbone_lr_mult": 0.3282, "weight_decay": 0.0356, "dropout": 0.3742, "mixup": 0.2777, "epochs": 51, "bg_prob": 0.5371, "aug_strong": true, "weight_alpha": 0.303, "aux_weight": 0.7602}`

  Stage 6: top 3 re-run with a second seed
  - trial 18: seed 0 mAP 0.319, seed 1 mAP 0.313 -> mean 0.316; F1 0.379 / 0.375
  - trial 1: seed 0 mAP 0.317, seed 1 mAP 0.312 -> mean 0.314; F1 0.379 / 0.382
  - trial 15: seed 0 mAP 0.317, seed 1 mAP 0.312 -> mean 0.314; F1 0.373 / 0.365
  - **resnet50 winner: trial 18, 10-model mean mAP 0.316.**
  (175 runs trained this session; remaining estimate ~0.8 h at 4.8 min/run)

### Backbone comparison (10-model mean val mAP)

- convnext_tiny: 0.326 (trial 8; F1 0.384 / 0.388; top type 0.427 / 0.426)
- resnet50: 0.316 (trial 18; F1 0.379 / 0.375; top type 0.429 / 0.419)
- resnet18: 0.307 (trial 15; F1 0.372 / 0.359; top type 0.411 / 0.404)

**Overall winner: convnext_tiny, trial 8.** Ablation: same settings with Black/White-style sprites.
- with BW style: mAP 0.324 vs 0.326 without; F1 0.387 vs 0.386

Round 3 sweep finished 2026-10-16 12:51. Final configuration in runs/r3/final.json.
