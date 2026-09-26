# Audio CNN — Setup Guide (Windows, Modal Starter)

Train on Modal's GPU once, download the model, then run the model **and** the dashboard entirely on your laptop, with no internet needed.

```
 Modal (cloud, one-time)                 Your laptop (offline)
 ┌──────────────────────────┐            ┌─────────────────────────────────────────┐
 │ train.py  →  esc-model   │  download  │ best_model.pth                          │
 │ (A10G GPU)   volume      │ ─────────► │ local_server.py  :8000/inference        │
 │              best_model  │            │        ▲                                │
 └──────────────────────────┘            │        │ POST base64 WAV                │
                                         │ Next.js dashboard  :3000                │
                                         └─────────────────────────────────────────┘
```

## What was fixed in the code

| # | Problem | Fix |
|---|---------|-----|
| 1 | torchaudio ≥ 2.9 makes `torchaudio.load()` need TorchCodec → `train.py` crashes | `requirements.txt` pins `torch==2.8.0`, `torchaudio==2.8.0` |
| 2 | `modal` not listed anywhere | New `requirements-local.txt` = `requirements.txt` + `modal` + `uvicorn`. Kept separate so the Modal images don't reinstall their own client |
| 3 | `main.py` f-strings only parse on Python 3.12+ | Rewritten to work on any Python 3.9+. Modal images pinned to `python_version="3.12"` so they no longer depend on your local version |
| 4 | `chirpingbirds.wav` missing | `modal run main.py` downloads an ESC-50 bird clip if it's missing. Use `--file your.wav` to test your own |
| 5 | `main.py` crashed if you hadn't trained yet | Volume is `create_if_missing`, and both the CLI entrypoint and the server say clearly "Run `modal run train.py` first" |
| 6 | `npm run build` failed on existing lint errors | Fixed the 7 ESLint errors in `page.tsx`, `FeatureMap.tsx`, `progress.tsx` (behaviour unchanged) |
| + | Inference URL was hardcoded `"inference_url_here"` | Now `NEXT_PUBLIC_INFERENCE_URL` (validated in `src/env.js`), **defaults to `http://localhost:8000/inference`** |
| + | No way to run offline | New `local_server.py`: same preprocessing and response as `main.py`, no Modal. GPU if available, else CPU |

## Prerequisites

- Python 3.10, 3.11 or 3.12 on your laptop. The Modal images always use 3.12 regardless. Check what you have with `py --list`
- Node.js 20 or 22 LTS
- A Modal account (Starter plan: $30/month free compute)

## Part A — Train on Modal (online, one-time)

Run from the repo root (`audio-cnn\`). Modal resolves `requirements.txt` relative to where you run it.

```powershell
cd C:\Users\akash\OneDrive\Desktop\Projects\audio-cnn\audio-cnn

py -3.11 -m venv .venv          # or -3.12 / -3.10, whichever `py --list` shows
.\.venv\Scripts\Activate.ps1

pip install -r requirements-local.txt
modal setup                      # browser login, one-time
```

Train (about 45–90 minutes on an A10G, roughly $1–2 of credits):

```powershell
modal run --detach train.py
```

- `--detach` keeps training running if you close the terminal. Follow progress at modal.com → Apps → `audio-cnn`.
- The first run builds the image and downloads ESC-50 (2,000 clips, 50 classes). Fold 5 is used for validation.
- The best checkpoint is saved to the `esc-model` volume as `best_model.pth`. Expect roughly 80%+ validation accuracy.

Optional cloud check (deploys a temporary endpoint and classifies a bird clip):

```powershell
modal run main.py                        # auto-downloads chirpingbirds.wav
modal run main.py --file some_clip.wav   # or your own WAV
```

## Part B — Download the model and prepare for offline use (online, one-time)

```powershell
# 1. Download the trained checkpoint (~85 MB) into the repo root
modal volume get esc-model best_model.pth .
#    re-downloading after retraining?  add --force

# 2. (Optional) download the training curves.
#    The destination folder must exist first, otherwise Modal writes every file
#    to the same path (Windows: "[Errno 13] Permission denied").
mkdir tb_logs
modal volume get esc-model tensorboard_logs tb_logs
tensorboard --logdir tb_logs    # open http://localhost:6006

# 3. Install and build the dashboard
cd audio-cnn-visualisation
npm i
npm run build
cd ..
```

`npm run build` matters for offline use. The dashboard's font (Geist, via `next/font/google`) is fetched from Google Fonts once and bundled into `.next/`, so `npm start` never needs the internet afterwards.

## Part C — Run everything locally (works offline)

Terminal 1 (model server, from the repo root):

```powershell
.\.venv\Scripts\Activate.ps1
python local_server.py
# Loading best_model.pth on cpu ...
# Model ready (50 classes, val acc 8x.xx%)
# Uvicorn running on http://127.0.0.1:8000
```

Terminal 2 (dashboard):

```powershell
cd audio-cnn-visualisation
npm start
```

Open **http://localhost:3000** and upload a `.wav` file.

- Health check: http://localhost:8000/health
- Speed: about 0.2–2 s per 5 s clip on CPU after warm-up. The first request is slower.
- Server options: `python local_server.py --model path\to\best_model.pth --port 8000`
- For hot-reload while editing UI code, use `npm run dev` instead of `npm start`. Offline it may warn about the font and use a fallback.

## Training v2 (ImageNet-pretrained, target 90%+)

`train_v2.py` fine-tunes the same `AudioCNN` from ImageNet ResNet-34 weights, with a corrected spectrogram frontend, stronger augmentation and EMA. The original `best_model.pth` is never overwritten.

```powershell
modal run --detach train_v2.py                          # train folds 1-4, test on fold 5 (~10 min, ~$0.20)
modal run --detach train_v2.py --folds 1,2,3,4,5        # optional 5-fold CV in parallel (~$1)
```

- The reported number is the **final-epoch EMA model** on the held-out fold (no best-epoch cherry-picking).
- Outputs: `esc-model:/runs/<run_name>/fold<k>/` → `model.pth`, `metrics.json`, `tensorboard/`.
- The run name is printed at the start (default `v2_<date>_<time>`).

Use it locally:

```powershell
modal volume get esc-model runs/<run_name>/fold5/model.pth best_model_v2.pth
python local_server.py --model best_model_v2.pth
```

The checkpoint carries its own spectrogram settings, so `local_server.py` and `main.py` apply them automatically. Old checkpoints keep using the legacy settings.

## Switching the dashboard between local and Modal

The dashboard reads `NEXT_PUBLIC_INFERENCE_URL` at **build time**. With no `.env`, it uses the local server.

To use the Modal endpoint instead:

```powershell
modal deploy main.py            # prints https://<workspace>--audio-cnn-inference-audioclassifier-inference.modal.run
```

Create `audio-cnn-visualisation\.env`:

```
NEXT_PUBLIC_INFERENCE_URL="https://<workspace>--audio-cnn-inference-audioclassifier-inference.modal.run"
```

Then run `npm run build` again (or restart `npm run dev`). Delete the line and rebuild to go back to local. Stop the cloud endpoint with `modal app stop audio-cnn-inference`. It scales to zero after 15 s idle anyway.

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `Model file not found: best_model.pth` | Run Part B step 1 from the repo root |
| `No trained model in the 'esc-model' volume yet` | Training hasn't finished or saved yet. Check the Modal dashboard |
| `AuthError: Token missing` | `modal setup` |
| Dashboard shows `Failed to fetch` | `local_server.py` isn't running, or is on a different port than `NEXT_PUBLIC_INFERENCE_URL` |
| `Could not process audio` (HTTP 400) | Upload a real WAV file. The dashboard only accepts `.wav` |
| Unicode or emoji errors in the Windows console | `$env:PYTHONUTF8=1` |
| Modal rejects `gpu="A10G"` | Change to `gpu="A10"` in `train.py` and `main.py` |
| `py -3.12`: "No suitable Python runtime found" | Use an installed version (`py --list`, e.g. `py -3.11`), or `winget install -e --id Python.Python.3.12` |
| `modal volume get … ./tb_logs`: `[Errno 13] Permission denied` | Create the folder first: `mkdir tb_logs`, then rerun |
| `Execution policy` blocks `Activate.ps1` | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |

## Cost notes (Modal Starter, $30/month free)

- A10 GPU ≈ $0.000306/s ≈ $1.10/h. One training run ≈ $1–2.
- Local inference costs nothing. Set a spending limit in Modal → Settings → Usage to be safe.
