# Training on the desktop (RTX 4090)

Run these on the desktop in **PowerShell** (Start → "PowerShell"; Administrator is not needed unless noted). Each step says what you should see. Total hands-on time is about 15 minutes plus training.

## 1. Bring over the data

On the laptop the bundle is at `PokeProject\transfer\pokeproject_training_data.zip` (219 MB: the manifest, species table, and the 96×96 images and masks). Copy it to the desktop's **Downloads** folder by USB drive, cloud drive, or network share.

## 2. Install Git and Python (skip what's already installed)

```powershell
winget install --id Git.Git -e
winget install --id Python.Python.3.11 -e
```

Then **close and reopen PowerShell** so it finds them, and check:

```powershell
git --version
py -3.11 --version
nvidia-smi
```

`nvidia-smi` should list the RTX 4090. If it errors, update the NVIDIA driver (GeForce Experience / NVIDIA app) first.

## 3. Get the code and the data

```powershell
cd $HOME\Desktop
git clone https://github.com/GraysonOhnstad/PokeProject
cd PokeProject
tar -xf $HOME\Downloads\pokeproject_training_data.zip
Test-Path data\manifest.csv, data\processed\96, data\processed\96_mask
```

The last line should print `True` three times.

## 4. Create the environment (CUDA build of PyTorch)

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

The last line should print `2.14.1+cu126 True NVIDIA GeForce RTX 4090`. PyTorch is installed first, from the CUDA index, so the second `pip install` keeps it instead of pulling the CPU-only build. (`cu130` also exists; use it instead of `cu126` only if `nvidia-smi` shows CUDA Version 13.0 or higher.)

Check that your prompt starts with `(.venv)` before the `pip` lines; if activation failed, the installs would go to the system Python instead.

If `Activate.ps1` is blocked ("running scripts is disabled"), run this once, then activate again:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Whenever you open a new PowerShell later: `cd $HOME\Desktop\PokeProject` then `.venv\Scripts\Activate.ps1`.

## 5. Smoke test (about a minute)

```powershell
python scripts/train.py --arch small_cnn --name smoke --limit 512 --epochs 2 --warmup 1
Remove-Item -Recurse runs\smoke
```

You should see `on cuda` in the first line and two epoch lines.

## 6. The four training runs

```powershell
python scripts/train.py --arch resnet18
python scripts/train.py --arch resnet18 --no-bw-style
python scripts/train.py --arch small_cnn
python scripts/train.py --arch small_cnn --no-bw-style
```

Each prints one line per epoch (`*` marks a new best val mAP) and stops early once val mAP stops improving for 10 epochs (at most 60 epochs). Results go to `runs\resnet18_bw`, `runs\resnet18_nobw`, `runs\small_cnn_bw`, `runs\small_cnn_nobw`.

To run all four unattended instead:

```powershell
python scripts/train.py --arch resnet18; python scripts/train.py --arch resnet18 --no-bw-style; python scripts/train.py --arch small_cnn; python scripts/train.py --arch small_cnn --no-bw-style
```

## 7. Calibrate each run (validation only)

```powershell
foreach ($r in "resnet18_bw", "resnet18_nobw", "small_cnn_bw", "small_cnn_nobw") { python scripts/calibrate.py --run $r }
```

## 8. Bring the results back to the laptop

Zip the runs folder:

```powershell
tar -a -cf $HOME\Desktop\pokeproject_runs.zip runs
```

Copy `pokeproject_runs.zip` back to the laptop the same way as step 1. On the laptop, the final evaluation and the app run from there, since they need the full-resolution source images that only the laptop has. Do **not** run `scripts/evaluate.py` on the desktop.

## Round 2: unattended tuning sweep (ResNet-18)

One command runs every round-2 experiment back to back (15 training runs, roughly 2.5–4 hours on the 4090) and makes the keep/discard decisions itself using rules fixed in advance. The test set is never touched.

**1. Prevent sleep.** Settings → System → Power (& battery) → Screen, sleep & hibernate timeouts → set "When plugged in, put my device to sleep after" to **Never** for the duration. Sleep pauses training; if a run gets interrupted instead, rerunning the command resumes (the interrupted run restarts, finished runs are kept).

**2. Update the code and check the 192 px data** (from the `PokeProject` folder):

Round 1's logs are now in the repo, so `git pull` would refuse to overwrite your local copies. Set the local runs folder aside, pull, then merge the weight files (not in git) back in:

```powershell
Rename-Item runs runs_round1
git pull
robocopy runs_round1 runs /E /XC /XN /XO
Remove-Item -Recurse runs_round1
Test-Path data\processed\192, data\processed\192_mask, runs\resnet18_nobw\summary.json, runs\resnet18_nobw\best.pt
```

`robocopy` only adds files the pull didn't bring (the `best.pt` weights); it prints a summary table, which is normal. The last line should print `True` four times (`runs\resnet18_nobw` is round 1's run, which the sweep compares against).

**3. Start the sweep:**

```powershell
.venv\Scripts\python scripts/sweep.py
```

It prints each run's epochs as before, plus a decision after each stage. If it stops for any reason (sleep, reboot, Ctrl+C), run the same command again: finished runs are skipped.

**4. When it says `Sweep finished`**, `runs\sweep_decisions.md` has every decision and `reports\round2_comparison.md` the full val comparison. Zip the runs and bring them back as before:

```powershell
tar -a -cf $HOME\Desktop\pokeproject_runs_round2.zip runs reports\round2_comparison.md
```

## Round 3: cross-validated sweep over three backbones

One command trains and scores every round-3 configuration (ResNet-18, ConvNeXt-Tiny, ResNet-50; structural tests, 20 Optuna trials each, top-3 re-runs; 5-fold cross-validation, no pruning): roughly **450 training runs, an estimated 25–35 hours**, so plan for several nights. It is fully resumable: stop it any time (Ctrl+C, reboot) and run the same command again; finished runs are never retrained. The test set is never read.

**1. Space and sleep.** Keep at least **10 GB free** on the drive (the sweep warns if it's lower) and leave sleep set to **Never**.

**2. Update the code.** Round 2's logs are now in the repo, so set your local runs aside during the pull as before (answer `n` if git asks "Unlink of file ... failed"):

```powershell
cd C:\Users\Grayson\Desktop\PokeProject
Rename-Item runs runs_local
git pull
robocopy runs_local runs /E /XC /XN /XO
Remove-Item -Recurse runs_local
```

**3. Install Optuna and build the pre-enlarged images** (about a minute; they make training several times faster):

```powershell
.venv\Scripts\python -m pip install optuna
.venv\Scripts\python scripts/make_upscaled.py
```

**4. Quick check (2 minutes):** one short training run, which also shows the per-epoch speed on the GPU:

```powershell
.venv\Scripts\python scripts/train3.py --fold 0 --set limit=2000 epochs=2 seed=99
Remove-Item -Recurse runs\r3
```

**5. Start the sweep:**

```powershell
.venv\Scripts\python scripts/sweep3.py
```

Progress and every decision are appended to `runs\r3\decisions.md` (open it any time to see how far it is); after each backbone it prints a time estimate for the rest. ConvNeXt-Tiny and ResNet-50 download their pretrained weights (~100 MB each) the first time.

**6. When it prints `Round 3 sweep finished`,** zip the results and bring them back:

```powershell
tar -a -cf $HOME\Desktop\pokeproject_round3.zip runs\r3 reports\round3_summary.md
```
