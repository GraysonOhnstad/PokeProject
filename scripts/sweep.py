"""Round-2 tuning of ResNet-18, all stages unattended (validation only).

    python scripts/sweep.py

Runs four stages back to back. Each stage trains a few variants on top of the settings
kept so far, calibrates them on val, and applies a rule fixed in advance: a variant
replaces the current settings only if its val mAP beats the current settings' by more
than MIN_GAIN (about twice the epoch-to-epoch noise seen in round 1); otherwise the
simpler current settings are kept. All tuning runs use --no-bw-style (the brief's main
setting). Decisions are appended to runs/sweep_decisions.md as they are made.

  1  input resolution: 192 re-rendered vs 96 enlarged to 192 (round 1 = the 96 baseline)
  2  learning rates: pretrained layers at 0.1x the head's rate; everything at 1e-4
  3  regularization: mixup; heavier dropout + weight decay; both
  4  confirmation: chosen settings x 3 seeds vs round-1 settings x 3 seeds (round 1's
     run is seed 0), plus the chosen settings with BW style x 3 seeds (ablation)

Finished runs are skipped, so after an interruption just run the same command again.
Ends by writing reports/round2_comparison.md (val only; the test set is never read).
"""

import json
import shlex
import statistics
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
DECISIONS = RUNS / "sweep_decisions.md"
REFERENCE = "resnet18_nobw"  # round 1: 96 px, lr 3e-4 everywhere, weight decay 1e-2, dropout 0.2
MIN_GAIN = 0.01

STAGES = [
    ("input resolution", {"res192": "--size 192", "up192": "--input-size 192"}),
    ("learning rates", {"blr0.1": "--backbone-lr-mult 0.1", "lr1e-4": "--lr 1e-4"}),
    ("regularization", {"mixup": "--mixup 0.4", "reg": "--dropout 0.5 --weight-decay 0.05",
                        "mixup_reg": "--mixup 0.4 --dropout 0.5 --weight-decay 0.05"}),
]


def log(text):
    print(text, flush=True)
    with open(DECISIONS, "a", encoding="utf-8") as f:
        f.write(text + "\n")


def train(name, flags):
    """Train (unless already done) and calibrate one run; return its best val mAP."""
    run_dir = RUNS / name
    if (run_dir / "summary.json").exists():
        print(f"skipping {name}: already trained", flush=True)
    else:
        cmd = [sys.executable, "scripts/train.py", "--arch", "resnet18", "--name", name, *shlex.split(flags)]
        print(f"\n>>> {' '.join(cmd)}", flush=True)
        subprocess.run(cmd, check=True, cwd=ROOT)
    if not (run_dir / "calibration.json").exists():
        subprocess.run([sys.executable, "scripts/calibrate.py", "--run", name], check=True, cwd=ROOT)
    (run_dir / "sweep.json").write_text(json.dumps({"flags": flags}))
    return json.loads((run_dir / "summary.json").read_text())["val"]["mAP"]


def main():
    RUNS.mkdir(exist_ok=True)
    log(f"\n## Sweep started {datetime.now():%Y-%m-%d %H:%M}\n")
    ref_map = json.loads((RUNS / REFERENCE / "summary.json").read_text())["val"]["mAP"]
    base, base_map, base_name = "", ref_map, REFERENCE
    log(f"Start: round-1 settings ({REFERENCE}), val mAP {ref_map:.3f}. Keep a change only if it adds > {MIN_GAIN}.")

    names = [REFERENCE]
    for i, (title, variants) in enumerate(STAGES, 1):
        log(f"\n### Stage {i}: {title} (current: `{base or 'round-1 settings'}`, val mAP {base_map:.3f})")
        results = {}
        for label, flags in variants.items():
            name = f"r2_s{i}_{label}"
            results[name] = (f"{base} {flags}".strip(), train(name, f"{base} {flags} --no-bw-style"))
            names.append(name)
            log(f"- {name}: `{results[name][0]}` -> val mAP {results[name][1]:.3f}")
        best = max(results, key=lambda n: results[n][1])
        if results[best][1] > base_map + MIN_GAIN:
            base, base_map, base_name = results[best][0], results[best][1], best
            log(f"- **Kept {best}** (+{base_map - ref_map:.3f} over round 1 so far).")
        else:
            log(f"- No variant beat {base_name} by > {MIN_GAIN}; keeping current settings.")

    log(f"\n### Stage 4: confirmation with seeds (final settings: `{base or 'round-1 settings'}`)")
    groups = {"final (no BW)": [], "round-1 settings (no BW)": [REFERENCE], "final with BW (ablation)": []}
    for seed in (0, 1, 2):
        groups["final (no BW)"].append(f"r2_final_s{seed}")
        train(f"r2_final_s{seed}", f"{base} --seed {seed} --no-bw-style")
        if seed:
            groups["round-1 settings (no BW)"].append(f"r2_ref_s{seed}")
            train(f"r2_ref_s{seed}", f"--seed {seed} --no-bw-style")
        groups["final with BW (ablation)"].append(f"r2_final_bw_s{seed}")
        train(f"r2_final_bw_s{seed}", f"{base} --seed {seed}")
    for group, runs in groups.items():
        maps = [json.loads((RUNS / r / "summary.json").read_text())["val"]["mAP"] for r in runs]
        names += [r for r in runs if r not in names]
        log(f"- {group}: val mAP {statistics.mean(maps):.3f} +/- {statistics.stdev(maps):.3f} "
            f"over seeds ({', '.join(f'{v:.3f}' for v in maps)})")

    subprocess.run([sys.executable, "scripts/compare_runs.py", *names,
                    "--out", "reports/round2_comparison.md"], check=True, cwd=ROOT)
    log(f"\nSweep finished {datetime.now():%Y-%m-%d %H:%M}. Final settings: `{base or 'round-1 settings'}`.")


if __name__ == "__main__":
    main()
