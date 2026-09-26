import torch
import torch.nn as nn
import torchaudio.transforms as T


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               3, stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        self.use_shortcut = stride != 1 or in_channels != out_channels
        if self.use_shortcut:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False), nn.BatchNorm2d(out_channels))

    def forward(self, x, fmap_dict=None, prefix=""):
        out = self.conv1(x)
        out = self.bn1(out)
        out = torch.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        shortcut = self.shortcut(x) if self.use_shortcut else x
        out_add = out + shortcut

        if fmap_dict is not None:
            fmap_dict[f"{prefix}.conv"] = out_add

        out = torch.relu(out_add)
        if fmap_dict is not None:
            fmap_dict[f"{prefix}.relu"] = out

        return out


class AudioCNN(nn.Module):
    def __init__(self, num_classes=50):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, 7, stride=2, padding=3, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(3, stride=2, padding=1))
        self.layer1 = nn.ModuleList([ResidualBlock(64, 64) for i in range(3)])
        self.layer2 = nn.ModuleList(
            [ResidualBlock(64 if i == 0 else 128, 128, stride=2 if i == 0 else 1) for i in range(4)])
        self.layer3 = nn.ModuleList(
            [ResidualBlock(128 if i == 0 else 256, 256, stride=2 if i == 0 else 1) for i in range(6)])
        self.layer4 = nn.ModuleList(
            [ResidualBlock(256 if i == 0 else 512, 512, stride=2 if i == 0 else 1) for i in range(3)])

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x, return_feature_maps=False):
        if not return_feature_maps:
            x = self.conv1(x)
            for block in self.layer1:
                x = block(x)
            for block in self.layer2:
                x = block(x)
            for block in self.layer3:
                x = block(x)
            for block in self.layer4:
                x = block(x)
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            x = self.dropout(x)
            x = self.fc(x)
            return x
        else:
            feature_maps = {}
            x = self.conv1(x)
            feature_maps["conv1"] = x

            for i, block in enumerate(self.layer1):
                x = block(x, feature_maps, prefix=f"layer1.block{i}")
            feature_maps["layer1"] = x

            for i, block in enumerate(self.layer2):
                x = block(x, feature_maps, prefix=f"layer2.block{i}")
            feature_maps["layer2"] = x

            for i, block in enumerate(self.layer3):
                x = block(x, feature_maps, prefix=f"layer3.block{i}")
            feature_maps["layer3"] = x

            for i, block in enumerate(self.layer4):
                x = block(x, feature_maps, prefix=f"layer4.block{i}")
            feature_maps["layer4"] = x

            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            x = self.dropout(x)
            x = self.fc(x)
            return x, feature_maps


# ---------------------------------------------------------------------------
# Spectrogram frontend shared by training (train_v2.py) and inference
# (main.py, local_server.py). Its settings are stored in each checkpoint under
# "frontend", so inference always matches what the model was trained on.
# ---------------------------------------------------------------------------

# Settings used by the original train.py (checkpoints without a "frontend" key).
# Note: sample_rate=22050 here does not match the 44.1 kHz audio; kept only so
# old checkpoints keep producing the same inputs they were trained on.
LEGACY_FRONTEND = {
    "sample_rate": 22050, "n_fft": 1024, "hop_length": 512, "n_mels": 128,
    "f_min": 0.0, "f_max": 11025.0, "top_db": None, "mean": None, "std": None,
}


class SpectrogramFrontend(nn.Module):
    """Waveform [B, 1, samples] at 44.1 kHz -> normalised log-mel [B, 1, n_mels, frames]."""

    def __init__(self, config=None):
        super().__init__()
        cfg = {**LEGACY_FRONTEND, **(config or {})}
        self.config = cfg
        self.mel = T.MelSpectrogram(
            sample_rate=cfg["sample_rate"], n_fft=cfg["n_fft"], hop_length=cfg["hop_length"],
            n_mels=cfg["n_mels"], f_min=cfg["f_min"], f_max=cfg["f_max"])
        self.to_db = T.AmplitudeToDB(top_db=cfg["top_db"])  # top_db clamps per clip for [B, 1, ...] input
        self.mean = cfg["mean"]
        self.std = cfg["std"]

    def forward(self, waveform):
        spec = self.to_db(self.mel(waveform))
        if self.mean is not None and self.std is not None:
            spec = (spec - self.mean) / self.std
        return spec


def load_imagenet_resnet34(model):
    """Initialise AudioCNN's backbone from torchvision's ImageNet ResNet-34.

    AudioCNN has the same layout as ResNet-34 (3-4-6-3 BasicBlocks), so every
    backbone tensor maps 1:1 by name. The first conv's RGB filters are summed
    into one channel (equivalent to feeding the spectrogram to all 3 channels).
    The classifier head is re-initialised for the new classes.
    """
    import torchvision  # only needed for training

    weights = torchvision.models.ResNet34_Weights.IMAGENET1K_V1
    source = torchvision.models.resnet34(weights=weights).state_dict()

    mapped = {}
    for key, value in source.items():
        if key.startswith("fc."):
            continue
        if key.startswith("conv1."):
            key = "conv1.0." + key[len("conv1."):]
        elif key.startswith("bn1."):
            key = "conv1.1." + key[len("bn1."):]
        mapped[key.replace(".downsample.", ".shortcut.")] = value
    mapped["conv1.0.weight"] = mapped["conv1.0.weight"].sum(dim=1, keepdim=True)

    missing, unexpected = model.load_state_dict(mapped, strict=False)
    if set(missing) != {"fc.weight", "fc.bias"} or unexpected:
        raise RuntimeError(f"ImageNet weight mapping failed: missing={missing}, unexpected={unexpected}")

    nn.init.normal_(model.fc.weight, std=0.01)
    nn.init.zeros_(model.fc.bias)
    return model
