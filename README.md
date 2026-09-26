# Audio CNN — Environmental Sound Classification

A ResNet-34-style convolutional network that classifies environmental sounds (dog bark, rain, chirping birds, siren, … 50 classes) from log-mel spectrograms, trained on **ESC-50**, with a Next.js dashboard that plays the audio and visualises the model's predictions, input spectrogram, waveform and internal feature maps.

**Headline result: 91.3% ± 1.3 accuracy (ESC-50, 5-fold cross-validation)**, up from 83.0% for the original from-scratch model.

- Training runs on [Modal](https://modal.com) serverless GPUs (A10G): about 2 minutes and a few cents per run.
- Inference runs **fully offline on your laptop** (CPU is enough), or optionally as a Modal web endpoint.

---

## Contents

1. [Results](#results)
2. [Model architecture](#model-architecture)
3. [Input frontend](#input-frontend-log-mel-spectrogram)
4. [Training configurations](#training-configurations)
5. [How to run](#how-to-run)
6. [Testing with ESC-50 clips](#testing-with-esc-50-clips)
7. [Project structure](#project-structure)
8. [Troubleshooting](#troubleshooting)
9. [Credits and license](#credits-and-license)

---

## Results

### Run log

| Run | Script | Evaluation | Accuracy | Train time (A10G) |
|---|---|---|---|---|
| Baseline | `train.py` | Fold 5, best epoch *(checkpoint picked on the test fold, so optimistic)* | 83.00% | ~48 min |
| `v2_20260926_223104` | `train_v2.py` | Fold 5, final-epoch EMA model | **90.75%** | 2.2 min |
| `v2_cv5` | `train_v2.py --folds 1,2,3,4,5` | 5-fold CV, final-epoch EMA model per fold | **91.30% ± 1.30** | ~2.2 min per fold (in parallel) |

Per-fold results for `v2_cv5`:

| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean ± std |
|---|---|---|---|---|---|
| 90.75% | 90.25% | 92.00% | 93.25% | 90.25% | **91.30% ± 1.30** |

Notes:
- **Evaluation protocol (v2):** train on 4 folds, test on the held-out fold, and report the **final-epoch** EMA model. No checkpoint is picked by looking at the test fold. The best epoch is logged for reference only; in the fold 5 run it reached 91.75% at epoch 49.
- Fold 5 scored 90.75% and 90.25% in two runs with the same seed. GPU non-determinism causes about ±0.5 points of run-to-run noise.
- ESC-50's 5 folds keep clips from the same source recording together, so there is no train/test leakage between folds.

### Context (published ESC-50 results)

| Reference point | Accuracy |
|---|---|
| Original ESC-50 CNN baseline (Piczak, 2015) | 64.5% |
| Human listeners | 81.3% |
| PANNs CNN14 trained from scratch (Kong et al., 2020) | 83.3% |
| **This project, baseline (from scratch)** | **83.0%** |
| ImageNet-pretrained ResNet (Palanisamy et al., 2020) | 90.65% |
| **This project, v2 (ImageNet-pretrained ResNet-34)** | **91.3% ± 1.3** |
| PANNs CNN14 fine-tuned from AudioSet | 94.7% |
| EfficientAT MobileNet, AudioSet-pretrained (Schmid et al.) | 96.4% |
| BEATs (current state of the art) | 98.1% |

### Model files

| File | Model | Accuracy |
|---|---|---|
| `best_model.pth` | Baseline, from scratch (`train.py`) | 83.0% (fold 5) |
| `best_model_v2.pth` | v2, fold 5 held out | 90.75% (fold 5) |
| `best_model_cv5.pth` | v2, fold 4 held out (best CV fold) | 93.25% (fold 4); the recipe's CV mean is 91.3% |

Model files (`*.pth`, about 85 MB each) are git-ignored. Download them from the Modal volume as shown in [How to run](#4-download-a-model-and-training-curves).

---

## Model architecture

`AudioCNN` in `model.py` has the same layout as ResNet-34 (BasicBlocks 3-4-6-3), adapted to a single-channel spectrogram input. **21.3 M parameters.**

```
Input: log-mel spectrogram  [B, 1, 128 mel bins, 431 frames]  (5 s clip)
│
├─ conv1   Conv 7×7, 64, stride 2 → BatchNorm → ReLU → MaxPool 3×3, stride 2
├─ layer1  3 × ResidualBlock(64)                    
├─ layer2  4 × ResidualBlock(128), first block stride 2
├─ layer3  6 × ResidualBlock(256), first block stride 2
├─ layer4  3 × ResidualBlock(512), first block stride 2
│
├─ AdaptiveAvgPool 1×1 → Dropout(0.5) → Linear(512 → 50)
Output: 50 class logits

ResidualBlock: Conv3×3 → BN → ReLU → Conv3×3 → BN → (+ shortcut) → ReLU
               shortcut = 1×1 Conv + BN when stride ≠ 1 or channels change, else identity
```

| Stage | Output shape (5 s clip) |
|---|---|
| conv1 | 64 × 32 × 108 |
| layer1 | 64 × 32 × 108 |
| layer2 | 128 × 16 × 54 |
| layer3 | 256 × 8 × 27 |
| layer4 | 512 × 4 × 14 |

With `return_feature_maps=True` the model also returns every stage and every block's `conv` (pre-activation sum) and `relu` output. The dashboard averages these over channels and shows them as heat-maps.

**v2 initialisation:** `load_imagenet_resnet34()` in `model.py` loads torchvision's ImageNet ResNet-34 weights. All 216 backbone tensors map one-to-one by name. The first conv's RGB filters are summed into one input channel, and the 50-class head is re-initialised (normal, std 0.01). The architecture itself is unchanged, so v1 and v2 checkpoints load into the same class.

---

## Input frontend (log-mel spectrogram)

Audio is mono and resampled to 44.1 kHz (ESC-50's native rate). `SpectrogramFrontend` in `model.py` converts it to a log-mel spectrogram. **v2 checkpoints store these settings (including the normalisation statistics)**, and `local_server.py` / `main.py` read them automatically, so inference always matches training. Checkpoints without them (the baseline) use the legacy settings.

| Setting | Legacy (baseline, `train.py`) | v2 (`train_v2.py`) |
|---|---|---|
| Sample rate given to MelSpectrogram | 22,050 (does not match the 44.1 kHz audio) | 44,100 (correct) |
| n_fft / hop length | 1024 / 512 | 1024 / 512 |
| Mel bins | 128 | 128 |
| Frequency range | 0 – 11,025 Hz nominal | 20 – 22,050 Hz |
| dB conversion | `AmplitudeToDB()`, no floor | `AmplitudeToDB(top_db=80)`, clamped per clip |
| Normalisation | none | (x − mean) / std, computed on the training folds |
| Output for a 5 s clip | 1 × 128 × 431 | 1 × 128 × 431 |

---

## Training configurations

| | Baseline `train.py` | v2 `train_v2.py` |
|---|---|---|
| Initialisation | PyTorch default (random) | **ImageNet-pretrained ResNet-34** |
| Epochs / batch size | 100 / 32 | 60 / 32 |
| Optimizer | AdamW, lr 5e-4, weight decay 0.01 on all params | AdamW, **backbone lr 3e-4, head lr 3e-3**, weight decay 0.05 (not on BN/bias) |
| LR schedule | OneCycle, max 2e-3, 10% warm-up | 3-epoch linear warm-up, then cosine to 0 |
| Loss | Cross-entropy, label smoothing 0.1 | Soft-target cross-entropy, label smoothing 0.1 |
| Mixup | Beta(0.2, 0.2) on ~30% of batches (spectrograms) | Beta(0.4, 0.4) on **every** batch (waveforms) |
| SpecAugment | 1 × freq mask (30), 1 × time mask (80) | 2 × freq mask (24), 2 × time mask (60) |
| Waveform augmentation | none | random circular time shift, random gain ±6 dB |
| Weight averaging | none | EMA, decay 0.998 (~10-epoch window) |
| Saved model | best epoch on fold 5 | final-epoch EMA model |
| Evaluation | fold 5 (also used for checkpoint selection) | held-out fold(s), optional 5-fold CV |
| Reproducibility | no seed | seed 42 |
| Data pipeline | per-clip ffmpeg decode + CPU spectrogram, 1 worker | all clips decoded once and kept on GPU; spectrograms and augmentation computed on GPU |
| Precision | fp32 | bf16 autocast, channels_last |
| Output location (Modal volume `esc-model`) | `/best_model.pth`, `/tensorboard_logs/` | `/runs/<run_name>/fold<k>/{model.pth, metrics.json, tensorboard/}` |

**Why v2 trains about 20× faster:**
- The old pipeline re-decoded every clip and rebuilt its spectrogram on one CPU core every epoch, including validation, which left the GPU idle.
- v2 decodes the audio once, does everything else on the GPU in bf16, and needs 60 epochs instead of 100.

---

## How to run

All commands are PowerShell on Windows, run from the repo root. macOS and Linux equivalents are noted where they differ.

### Prerequisites

| Tool | Version | Notes |
|---|---|---|
| Python | 3.10 – 3.12 | The Modal images always use 3.12 regardless of your local version |
| Node.js | 20, 22 or 24 LTS | **Not Node 25** (breaks Next.js 15 dev server, see [Troubleshooting](#troubleshooting)) |
| Modal account | Starter plan is enough | $30/month free compute; a v2 run costs about $0.05 |
| Git | any | |

### 1. Clone and install

```powershell
git clone https://github.com/Akash-Sakala/audio-cnn.git
cd audio-cnn

py -3.11 -m venv .venv                 # macOS/Linux: python3 -m venv .venv
.\.venv\Scripts\Activate.ps1           # macOS/Linux: source .venv/bin/activate

pip install -r requirements-local.txt  # requirements.txt + modal + uvicorn
modal setup                            # one-time browser login
```

### 2. Train (on Modal)

**v2, recommended** (ImageNet-pretrained, about 2 min per fold):

```powershell
# Train on folds 1-4, test on fold 5
modal run --detach train_v2.py --run-name my_run

# 5-fold cross-validation, all folds in parallel
modal run --detach train_v2.py --folds 1,2,3,4,5 --run-name v2_cv5

# Other options
modal run --detach train_v2.py --epochs 80 --seed 7 --run-name my_run_80ep
```

- The first run builds the image, which downloads ESC-50 and the ImageNet weights once and takes a few extra minutes.
- `--detach` keeps training running if you close the terminal.
- The run prints a summary with each fold's final accuracy and, for several folds, the mean ± std.
- Progress is also visible at modal.com → Apps → `audio-cnn-v2`.

**Baseline** (original recipe, from scratch, about 48 min):

```powershell
modal run --detach train.py            # writes /best_model.pth on the volume
```

### 3. Check a run

```powershell
modal volume ls esc-model runs/v2_cv5
modal volume ls esc-model runs/v2_cv5/fold4
```

### 4. Download a model and training curves

```powershell
# A v2 model (one per fold)
modal volume get esc-model runs/v2_cv5/fold4/model.pth best_model_cv5.pth

# The baseline model
modal volume get esc-model best_model.pth .
```

Add `--force` to overwrite an existing local file.

**Scores and curves for all folds of a CV run.** Folders must exist before downloading a directory:

```powershell
1..5 | ForEach-Object {
    mkdir "cv5/fold$_/tb" -Force | Out-Null
    modal volume get esc-model "runs/v2_cv5/fold$_/metrics.json" "cv5/fold$_/metrics.json"
    modal volume get esc-model "runs/v2_cv5/fold$_/tensorboard" "cv5/fold$_/tb"
}

# Print per-fold accuracy and the mean
$accs = 1..5 | ForEach-Object { (Get-Content "cv5/fold$_/metrics.json" | ConvertFrom-Json).final_ema_acc }
1..5 | ForEach-Object { "fold $_ : {0:N2}%" -f $accs[$_-1] }
"mean   : {0:N2}%" -f ($accs | Measure-Object -Average).Average

# Compare training curves (http://localhost:6006)
tensorboard --logdir cv5
```

Each `metrics.json` holds the final accuracy, the best epoch (reference only), the full config and per-epoch history (loss, validation accuracy for the raw and EMA models, learning rate).

### 5. Run the app locally (offline)

Everything below works without internet once the model is downloaded and the dashboard is built.

**Terminal 1: model server** (FastAPI on port 8000, uses a GPU if available, otherwise CPU):

```powershell
.\.venv\Scripts\Activate.ps1
python local_server.py --model best_model_cv5.pth
# Model ready (50 classes, val acc 93.25%, frontend: v2 (saved in checkpoint))
# Uvicorn running on http://127.0.0.1:8000
```

- Health check: http://localhost:8000/health
- Options: `--model <path>` (default `best_model.pth`), `--host`, `--port` (default 8000)
- Endpoints: `POST /inference` with `{"audio_data": "<base64 WAV>"}`, and `GET /health`

**Terminal 2: dashboard** (Next.js on port 3000):

```powershell
cd audio-cnn-visualisation
npm ci                  # first time only
npm run build           # first time, and after code changes; bundles the font for offline use
npm start               # or: npm run dev (hot reload)
```

Open **http://localhost:3000** and choose a `.wav` file. The dashboard shows:
- an **audio player**, available as soon as the file is chosen
- the **top 3 predictions** with confidence
- the **input spectrogram** and **waveform**
- **feature maps** for every layer and residual block

The dashboard calls `http://localhost:8000/inference` by default. To point it elsewhere, set `NEXT_PUBLIC_INFERENCE_URL` in `audio-cnn-visualisation/.env` and rebuild; the value is read at build time.

### 6. (Optional) Serve the model from Modal instead

`main.py` deploys a GPU web endpoint that loads `/best_model.pth` from the volume. To serve a v2 model, back up the baseline and upload the new file under that name:

```powershell
modal volume cp esc-model best_model.pth best_model_baseline.pth      # keep the 83% model
modal volume put esc-model best_model_cv5.pth best_model.pth --force  # upload the v2 model
modal run main.py                      # test: classifies an ESC-50 bird clip (auto-downloaded)
modal deploy main.py                   # prints the endpoint URL
```

Then create `audio-cnn-visualisation/.env` with `NEXT_PUBLIC_INFERENCE_URL="<printed URL>"`, rebuild the dashboard, and run it. The endpoint scales to zero after 15 s idle. Stop it with `modal app stop audio-cnn-inference`.

---

## Testing with ESC-50 clips

Download one clip per class from the fold the model **did not** train on. That is fold 4 for `best_model_cv5.pth` and fold 5 for `best_model_v2.pth`.

```powershell
$ProgressPreference = "SilentlyContinue"
$fold = 4; $perClass = 1
$base = "https://raw.githubusercontent.com/karolpiczak/ESC-50/master"
$outDir = "test_clips_fold$fold"
mkdir $outDir -Force | Out-Null
Invoke-WebRequest "$base/meta/esc50.csv" -OutFile "$outDir/esc50.csv"
$picks = Import-Csv "$outDir/esc50.csv" | Where-Object { $_.fold -eq "$fold" } |
         Group-Object category | ForEach-Object { $_.Group | Select-Object -First $perClass }
foreach ($p in $picks) {
    $out = "$outDir/$($p.category)__$($p.filename)"
    if (-not (Test-Path $out)) { Invoke-WebRequest "$base/audio/$($p.filename)" -OutFile $out }
}
```

**Batch-check every clip against the running local server:**

```powershell
$files = Get-ChildItem $outDir -Filter *.wav; $correct = 0
foreach ($f in $files) {
    $truth = ($f.Name -split '__')[0]
    $body  = @{ audio_data = [Convert]::ToBase64String([IO.File]::ReadAllBytes($f.FullName)) } | ConvertTo-Json
    $pred  = (Invoke-RestMethod http://localhost:8000/inference -Method Post -ContentType application/json -Body $body).predictions[0]
    if ($pred.class -eq $truth) { $correct++ }
    "{0,-18} -> {1,-18} {2,6:P1}" -f $truth, $pred.class, $pred.confidence
}
"Accuracy: $correct / $($files.Count)"
```

---

## Project structure

```
audio-cnn/
├── model.py                  # AudioCNN, SpectrogramFrontend, ImageNet weight loader
├── train.py                  # Baseline training (from scratch) on Modal
├── train_v2.py               # v2 training: ImageNet init, GPU pipeline, EMA, k-fold CV
├── main.py                   # Modal GPU inference endpoint (FastAPI)
├── local_server.py           # Offline local inference server (FastAPI, CPU/GPU)
├── requirements.txt          # Python deps, also installed into the Modal images
├── requirements-local.txt    # + modal CLI, uvicorn (for your machine)
├── setup-guide.md            # Detailed Windows setup notes
├── theory.excalidraw         # Architecture/theory diagram
└── audio-cnn-visualisation/  # Next.js 15 + React 19 + Tailwind dashboard
    ├── src/app/page.tsx      # Upload, audio player, predictions, visualisations
    ├── src/components/       # FeatureMap, Waveform, ColorScale, UI components
    └── src/env.js            # NEXT_PUBLIC_INFERENCE_URL (default: local server)
```

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| Dashboard: `Failed to fetch` | The model server isn't running, or it's on another port. Start `python local_server.py` and check http://localhost:8000/health |
| `npm run dev`: `localStorage.getItem is not a function` | You're on Node 25. Use Node 24 LTS (`nvm install 24; nvm use 24`), or run `$env:NODE_OPTIONS="--no-experimental-webstorage"` first |
| `npm run build`: `<Html> should not be imported outside of pages/_document` | A global `NODE_ENV` is set. Run `Remove-Item Env:NODE_ENV` and rebuild |
| `modal volume get … <folder>`: `[Errno 13] Permission denied` | Create the destination folder first (`mkdir <folder>`) |
| `Model file not found` | Download a model first (step 4) |
| `AuthError: Token missing` | `modal setup` |
| `py -3.12`: "No suitable Python runtime found" | Use an installed version (`py --list`), e.g. `py -3.11` |
| `Activate.ps1` blocked by execution policy | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |
| Unicode or emoji errors in the Windows console | `$env:PYTHONUTF8=1` |
| Modal rejects `gpu="A10G"` | Change it to `gpu="A10"` in the training and inference scripts |

**Costs (Modal Starter, $30/month free):**
- An A10 GPU costs about $1.10/hour.
- A v2 run is about $0.05 per fold; a full 5-fold CV about $0.20; the baseline run about $1.
- Local inference is free.

---

## Credits and license

- Based on [Andreas Trolle's audio-cnn](https://github.com/Andreaswt/audio-cnn) (MIT). This fork adds the v2 training recipe, 5-fold evaluation, offline local inference, frontend audio playback, and setup fixes.
- Dataset: [ESC-50](https://github.com/karolpiczak/ESC-50) by Karol J. Piczak (CC BY-NC 3.0).
- ImageNet weights: [torchvision ResNet-34](https://pytorch.org/vision/stable/models/resnet.html).
- References: [PANNs (Kong et al., 2020)](https://arxiv.org/abs/1912.10211), [Rethinking CNN Models for Audio Classification (Palanisamy et al., 2020)](https://arxiv.org/abs/2007.11154), [EfficientAT (Schmid et al.)](https://github.com/fschmid56/EfficientAT), [SpecAugment (Park et al., 2019)](https://arxiv.org/abs/1904.08779).

Licensed under the MIT License, see [LICENSE.MD](LICENSE.MD).
