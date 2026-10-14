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
