"""
Audio CNN v2 training: fine-tune an ImageNet-pretrained ResNet-34 on ESC-50.

Same AudioCNN architecture as train.py (so main.py, local_server.py and the
dashboard work unchanged), but with a much better recipe:

  1. ImageNet-pretrained ResNet-34 initialisation (conv1 RGB -> 1 channel)
  2. Differential learning rates (backbone 3e-4, head 3e-3), warmup + cosine,
     weight decay 0.05 excluding BatchNorm and bias
  3. Correct log-mel frontend (44.1 kHz, top_db=80) + train-set mean/std
     normalisation, saved in the checkpoint for inference
  4. Mixup on every batch (Beta 0.4), 2x freq + 2x time masks, random circular
     time shift, random gain +-6 dB
  5. EMA of weights (decay 0.998, ~10-epoch window); the final EMA model is saved
  6. Honest evaluation: fixed seed, report the final-epoch EMA model on the
     held-out fold (no best-epoch cherry-picking); optional 5-fold CV
  7. All audio preloaded on the GPU; spectrograms/augmentation on the GPU; bf16
  8. Each run saved under /models/runs/<run_name>/fold<k>/ (best_model.pth untouched)

Usage (from the repo root, venv active):
    modal run --detach train_v2.py                         # train folds 1-4, test fold 5
    modal run --detach train_v2.py --folds 1,2,3,4,5       # 5-fold CV, folds run in parallel
    modal run --detach train_v2.py --epochs 80 --run-name my_experiment

Download the result:
    modal volume get esc-model runs/<run_name>/fold5/model.pth best_model_v2.pth
    python local_server.py --model best_model_v2.pth
"""
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import modal

app = modal.App("audio-cnn-v2")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install(["wget", "unzip", "libsndfile1"])
    .pip_install_from_requirements("requirements.txt")
    .pip_install("torchvision==0.23.0")  # matches torch 2.8.0; only used to fetch ImageNet weights
    .run_commands([
        "cd /tmp && wget -q https://github.com/karolpiczak/ESC-50/archive/master.zip -O esc50.zip",
        "cd /tmp && unzip -q esc50.zip",
        "mkdir -p /opt/esc50-data",
        "cp -r /tmp/ESC-50-master/* /opt/esc50-data/",
        "rm -rf /tmp/esc50.zip /tmp/ESC-50-master",
        # Bake the ImageNet ResNet-34 weights into the image (no download per run)
        "python -c \"import torchvision; torchvision.models.resnet34(weights='IMAGENET1K_V1')\"",
    ])
    .add_local_python_source("model")
)

model_volume = modal.Volume.from_name("esc-model", create_if_missing=True)

ESC50_DIR = "/opt/esc50-data"
RUNS_DIR = "/models/runs"
CLIP_SAMPLES = 220_500  # ESC-50: 5 s mono at 44.1 kHz


@dataclass
class Config:
    # optimisation
    epochs: int = 60
    batch_size: int = 32
    lr_backbone: float = 3e-4
    lr_head: float = 3e-3
    weight_decay: float = 0.05
    warmup_epochs: int = 3
    label_smoothing: float = 0.1
    ema_decay: float = 0.998  # ~500-step (~10 epoch) averaging window for a 3000-step run
    pretrained: bool = True
    seed: int = 42
    # augmentation
    mixup_alpha: float = 0.4
    freq_masks: int = 2
    freq_mask_param: int = 24
    time_masks: int = 2
    time_mask_param: int = 60
    max_gain_db: float = 6.0
    time_shift: bool = True
    # frontend (waveforms are 44.1 kHz)
    sample_rate: int = 44_100
    n_fft: int = 1024
    hop_length: int = 512
    n_mels: int = 128
    f_min: float = 20.0
    f_max: float = 22_050.0
    top_db: float = 80.0
    # debugging: cap clips per split (0 = all); used for quick CPU smoke tests
    limit: int = 0


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_esc50(data_dir: str, test_fold: int, limit: int = 0):
    """Return (train_waves, train_labels, test_waves, test_labels, classes) as CPU tensors."""
    from concurrent.futures import ThreadPoolExecutor

    import numpy as np
    import pandas as pd
    import soundfile as sf
    import torch

    data_dir = Path(data_dir)
    meta = pd.read_csv(data_dir / "meta" / "esc50.csv")
    classes = sorted(meta["category"].unique())  # same ordering as train.py
    class_to_idx = {c: i for i, c in enumerate(classes)}

    def read(filename: str) -> np.ndarray:
        audio, sr = sf.read(data_dir / "audio" / filename, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != 44_100:
            raise ValueError(f"{filename}: expected 44.1 kHz, got {sr}")
        out = np.zeros(CLIP_SAMPLES, dtype=np.float32)
        out[: min(len(audio), CLIP_SAMPLES)] = audio[:CLIP_SAMPLES]
        return out

    def split(rows):
        if limit:
            rows = rows.sample(n=min(limit, len(rows)), random_state=0)
        with ThreadPoolExecutor(max_workers=16) as pool:
            waves = list(pool.map(read, rows["filename"]))
        x = torch.from_numpy(np.stack(waves)).unsqueeze(1)  # [N, 1, samples]
        y = torch.tensor([class_to_idx[c] for c in rows["category"]], dtype=torch.long)
        return x, y

    x_tr, y_tr = split(meta[meta["fold"] != test_fold])
    x_te, y_te = split(meta[meta["fold"] == test_fold])
    return x_tr, y_tr, x_te, y_te, classes


# ---------------------------------------------------------------------------
# GPU augmentation helpers
# ---------------------------------------------------------------------------
def random_time_shift(x):
    """Circularly shift each clip by a random offset. x: [B, 1, T]."""
    import torch

    b, _, t = x.shape
    shifts = torch.randint(0, t, (b, 1), device=x.device)
    idx = (torch.arange(t, device=x.device).unsqueeze(0) + shifts) % t
    return torch.gather(x.squeeze(1), 1, idx).unsqueeze(1)


def random_gain(x, max_db: float):
    import torch

    gain_db = (torch.rand(x.shape[0], 1, 1, device=x.device) * 2 - 1) * max_db
    return x * torch.pow(10.0, gain_db / 20.0)


def mixup(x, y_onehot, alpha: float):
    import numpy as np
    import torch

    lam = float(np.random.beta(alpha, alpha))
    perm = torch.randperm(x.shape[0], device=x.device)
    return lam * x + (1 - lam) * x[perm], lam * y_onehot + (1 - lam) * y_onehot[perm]


# ---------------------------------------------------------------------------
# Training (plain function so it can also run on CPU for smoke tests)
# ---------------------------------------------------------------------------
def run_training(test_fold: int, run_dir: str, cfg: Config, data_dir: str = ESC50_DIR) -> dict:
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torchaudio.transforms as T
    from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
    from torch.utils.tensorboard import SummaryWriter

    from model import AudioCNN, SpectrogramFrontend, load_imagenet_resnet34

    # ---- setup ---------------------------------------------------------------
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    torch.backends.cudnn.benchmark = True
    out_dir = Path(run_dir) / f"fold{test_fold}"
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(out_dir / "tensorboard"))
    t0 = time.time()

    # ---- data (preloaded once, kept on the device) ---------------------------
    x_tr, y_tr, x_te, y_te, classes = load_esc50(data_dir, test_fold, cfg.limit)
    x_tr, y_tr, x_te, y_te = x_tr.to(device), y_tr.to(device), x_te.to(device), y_te.to(device)
    num_classes = len(classes)
    print(f"[fold {test_fold}] train {len(x_tr)} clips, test {len(x_te)} clips, "
          f"{num_classes} classes, device {device}, bf16 {use_bf16}, "
          f"load {time.time() - t0:.0f}s", flush=True)

    frontend_cfg = {k: getattr(cfg, k) for k in
                    ("sample_rate", "n_fft", "hop_length", "n_mels", "f_min", "f_max", "top_db")}

    # Normalisation statistics from the (un-augmented) training set only
    raw_frontend = SpectrogramFrontend({**frontend_cfg, "mean": None, "std": None}).to(device)
    with torch.no_grad():
        total, total_sq, count = 0.0, 0.0, 0
        for i in range(0, len(x_tr), 100):
            s = raw_frontend(x_tr[i:i + 100]).double()
            total += s.sum().item()
            total_sq += (s * s).sum().item()
            count += s.numel()
    mean = total / count
    std = math.sqrt(max(total_sq / count - mean * mean, 1e-12))
    frontend_cfg.update(mean=mean, std=std)
    frontend = SpectrogramFrontend(frontend_cfg).to(device)
    print(f"[fold {test_fold}] log-mel mean {mean:.2f} dB, std {std:.2f} dB", flush=True)

    freq_mask = T.FrequencyMasking(cfg.freq_mask_param, iid_masks=True)
    time_mask = T.TimeMasking(cfg.time_mask_param, iid_masks=True)

    with torch.no_grad():  # test spectrograms are deterministic: compute once
        test_specs = torch.cat([frontend(x_te[i:i + 100]) for i in range(0, len(x_te), 100)])

    # ---- model -----------------------------------------------------------------
    model = AudioCNN(num_classes=num_classes)
    if cfg.pretrained:
        load_imagenet_resnet34(model)
    model = model.to(device).to(memory_format=torch.channels_last)
    ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(cfg.ema_decay), use_buffers=True)

    def param_groups():
        groups = {k: [] for k in ("bb_decay", "bb_no_decay", "head_decay", "head_no_decay")}
        for name, p in model.named_parameters():
            part = "head" if name.startswith("fc.") else "bb"
            kind = "no_decay" if p.ndim <= 1 else "decay"  # BatchNorm weights/biases and all biases
            groups[f"{part}_{kind}"].append(p)
        return [
            {"params": groups["bb_decay"], "lr": cfg.lr_backbone, "weight_decay": cfg.weight_decay},
            {"params": groups["bb_no_decay"], "lr": cfg.lr_backbone, "weight_decay": 0.0},
            {"params": groups["head_decay"], "lr": cfg.lr_head, "weight_decay": cfg.weight_decay},
            {"params": groups["head_no_decay"], "lr": cfg.lr_head, "weight_decay": 0.0},
        ]

    optimizer = torch.optim.AdamW(param_groups())
    steps_per_epoch = math.ceil(len(x_tr) / cfg.batch_size)
    total_steps = cfg.epochs * steps_per_epoch
    warmup_steps = cfg.warmup_epochs * steps_per_epoch

    def lr_factor(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)

    def evaluate(net) -> tuple[float, float]:
        net.eval()
        correct, loss_sum = 0, 0.0
        with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_bf16):
            for i in range(0, len(test_specs), 100):
                xb = test_specs[i:i + 100].contiguous(memory_format=torch.channels_last)
                logits = net(xb).float()
                loss_sum += F.cross_entropy(logits, y_te[i:i + 100], reduction="sum").item()
                correct += (logits.argmax(1) == y_te[i:i + 100]).sum().item()
        return 100.0 * correct / len(y_te), loss_sum / len(y_te)

    # ---- train -----------------------------------------------------------------
    history, best = [], {"acc": -1.0, "epoch": -1, "which": ""}
    step = 0
    for epoch in range(cfg.epochs):
        model.train()
        epoch_start, loss_sum = time.time(), 0.0
        order = torch.randperm(len(x_tr), device=device)
        for b in range(steps_per_epoch):
            idx = order[b * cfg.batch_size:(b + 1) * cfg.batch_size]
            if len(idx) < 2:
                continue
            with torch.no_grad():
                wave = x_tr[idx]
                if cfg.time_shift:
                    wave = random_time_shift(wave)
                if cfg.max_gain_db > 0:
                    wave = random_gain(wave, cfg.max_gain_db)
                target = F.one_hot(y_tr[idx], num_classes).float()
                if cfg.mixup_alpha > 0:
                    wave, target = mixup(wave, target, cfg.mixup_alpha)
                spec = frontend(wave)  # [B, 1, n_mels, frames], normalised
                for _ in range(cfg.freq_masks):
                    spec = freq_mask(spec)
                for _ in range(cfg.time_masks):
                    spec = time_mask(spec)
                if cfg.label_smoothing > 0:
                    target = target * (1 - cfg.label_smoothing) + cfg.label_smoothing / num_classes
                spec = spec.contiguous(memory_format=torch.channels_last)

            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_bf16):
                logits = model(spec)
            loss = torch.sum(-target * F.log_softmax(logits.float(), dim=1), dim=1).mean()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            ema.update_parameters(model)
            loss_sum += loss.item()
            step += 1

        train_loss = loss_sum / steps_per_epoch
        acc, val_loss = evaluate(model)
        ema_acc, ema_val_loss = evaluate(ema.module)
        for which, a in (("model", acc), ("ema", ema_acc)):
            if a > best["acc"]:
                best = {"acc": a, "epoch": epoch + 1, "which": which}
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss,
                        "val_acc": acc, "ema_val_loss": ema_val_loss, "ema_val_acc": ema_acc,
                        "lr_backbone": optimizer.param_groups[0]["lr"]})
        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("Loss/Validation", val_loss, epoch)
        writer.add_scalar("Loss/Validation_EMA", ema_val_loss, epoch)
        writer.add_scalar("Accuracy/Validation", acc, epoch)
        writer.add_scalar("Accuracy/Validation_EMA", ema_acc, epoch)
        writer.add_scalar("Learning_Rate/backbone", optimizer.param_groups[0]["lr"], epoch)
        print(f"[fold {test_fold}] epoch {epoch + 1:3d}/{cfg.epochs}  loss {train_loss:.4f}  "
              f"val {acc:5.2f}%  ema {ema_acc:5.2f}%  ({time.time() - epoch_start:.1f}s)", flush=True)

    # ---- save the final EMA model (the reported number) ---------------------------
    final_acc, final_loss = evaluate(ema.module)
    ema_state = {k: v.detach().cpu().contiguous() for k, v in ema.module.state_dict().items()}
    checkpoint = {
        "model_state_dict": ema_state,
        "classes": classes,
        "accuracy": final_acc,
        "epoch": cfg.epochs,
        "fold": test_fold,
        "frontend": frontend_cfg,
        "config": asdict(cfg),
        "recipe": "v2: ImageNet ResNet-34 fine-tune, EMA final model",
    }
    torch.save(checkpoint, out_dir / "model.pth")
    result = {
        "fold": test_fold,
        "final_ema_acc": final_acc,
        "final_ema_loss": final_loss,
        "best_epoch_acc_reference_only": best,
        "minutes": round((time.time() - t0) / 60, 1),
        "config": asdict(cfg),
        "history": history,
    }
    (out_dir / "metrics.json").write_text(json.dumps(result, indent=2))
    writer.close()
    print(f"[fold {test_fold}] FINAL (EMA, last epoch): {final_acc:.2f}%  |  best epoch seen: "
          f"{best['acc']:.2f}% @ {best['epoch']} ({best['which']}, optimistic, reference only)  |  "
          f"{result['minutes']} min", flush=True)
    return {k: v for k, v in result.items() if k != "history"}


# ---------------------------------------------------------------------------
# Modal wiring
# ---------------------------------------------------------------------------
@app.function(image=image, gpu="A10G", volumes={"/models": model_volume},
              timeout=60 * 60 * 2, memory=8192)
def train_fold(test_fold: int, run_name: str, cfg_dict: dict) -> dict:
    result = run_training(test_fold, f"{RUNS_DIR}/{run_name}", Config(**cfg_dict))
    model_volume.commit()
    return result


@app.local_entrypoint()
def main(folds: str = "5", run_name: str = "", epochs: int = 60, seed: int = 42):
    fold_list = [int(f) for f in folds.split(",") if f.strip()]
    if not fold_list or any(f not in range(1, 6) for f in fold_list):
        raise SystemExit("--folds must be a comma-separated list of 1..5, e.g. --folds 5 or --folds 1,2,3,4,5")
    run_name = run_name or time.strftime("v2_%Y%m%d_%H%M%S")
    cfg = asdict(Config(epochs=epochs, seed=seed))
    print(f"Run '{run_name}': folds {fold_list}, {epochs} epochs -> volume esc-model:/runs/{run_name}/")

    results = list(train_fold.starmap([(f, run_name, cfg) for f in fold_list]))

    print("\n==== Summary (final-epoch EMA model on each held-out fold) ====")
    for r in sorted(results, key=lambda r: r["fold"]):
        print(f"  fold {r['fold']}: {r['final_ema_acc']:.2f}%   ({r['minutes']} min)")
    accs = [r["final_ema_acc"] for r in results]
    if len(accs) > 1:
        m = sum(accs) / len(accs)
        s = (sum((a - m) ** 2 for a in accs) / (len(accs) - 1)) ** 0.5
        print(f"  mean: {m:.2f}% +- {s:.2f}")
    shown = max(r["fold"] for r in results)
    print(f"\nDownload: modal volume get esc-model runs/{run_name}/fold{shown}/model.pth best_model_v2.pth")
