"""
Offline inference server for the Audio CNN — runs on your laptop, no Modal needed.

Same request/response format as the Modal endpoint in main.py, so the
Next.js dashboard works unchanged once it points at this server.

Usage (from the repo root, venv active):
    python local_server.py                          # uses ./best_model.pth
    python local_server.py --model path/to/best_model.pth --port 8000

The dashboard calls http://localhost:8000/inference by default
(override with NEXT_PUBLIC_INFERENCE_URL in audio-cnn-visualisation/.env).
"""
import argparse
import base64
import io
import os

import librosa
import numpy as np
import soundfile as sf
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from model import AudioCNN, SpectrogramFrontend

TARGET_SR = 44100          # main.py resamples everything to 44.1 kHz (ESC-50's native rate)
MAX_WAVEFORM_POINTS = 8000


class AudioProcessor:
    """Uses the frontend settings saved in the checkpoint (legacy settings if absent)."""

    def __init__(self, frontend: dict | None = None):
        self.frontend = SpectrogramFrontend(frontend).eval()

    def process_audio_chunk(self, audio_data: np.ndarray) -> torch.Tensor:
        waveform = torch.from_numpy(audio_data).float().view(1, 1, -1)
        with torch.no_grad():
            return self.frontend(waveform)  # [1, 1, n_mels, time]


class InferenceRequest(BaseModel):
    audio_data: str  # base64-encoded WAV bytes


class AudioClassifier:
    def __init__(self, model_path: str):
        if not os.path.exists(model_path):
            raise SystemExit(
                f"Model file not found: {model_path}\n"
                "Train on Modal (`modal run train.py`), then download it with:\n"
                "    modal volume get esc-model best_model.pth .")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading {model_path} on {self.device} ...")
        checkpoint = torch.load(model_path, map_location=self.device)
        self.classes = checkpoint["classes"]
        self.model = AudioCNN(num_classes=len(self.classes))
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device).eval()
        self.audio_processor = AudioProcessor(checkpoint.get("frontend"))
        frontend = "v2 (saved in checkpoint)" if checkpoint.get("frontend") else "legacy"
        print(f"Model ready ({len(self.classes)} classes, "
              f"val acc {checkpoint.get('accuracy', float('nan')):.2f}%, frontend: {frontend})")

    def predict(self, audio_b64: str) -> dict:
        audio_bytes = base64.b64decode(audio_b64)
        audio_data, sample_rate = sf.read(io.BytesIO(audio_bytes), dtype="float32")

        if audio_data.ndim > 1:
            audio_data = np.mean(audio_data, axis=1)
        if sample_rate != TARGET_SR:
            audio_data = librosa.resample(y=audio_data, orig_sr=sample_rate, target_sr=TARGET_SR)

        spectrogram = self.audio_processor.process_audio_chunk(audio_data).to(self.device)

        with torch.no_grad():
            output, feature_maps = self.model(spectrogram, return_feature_maps=True)
            probabilities = torch.softmax(torch.nan_to_num(output), dim=1)
            top3_probs, top3_idx = torch.topk(probabilities[0], 3)

        predictions = [{"class": self.classes[i.item()], "confidence": p.item()}
                       for p, i in zip(top3_probs, top3_idx)]

        viz_data = {}
        for name, tensor in feature_maps.items():
            if tensor.dim() == 4:  # [batch, channels, H, W] -> mean over channels
                arr = np.nan_to_num(torch.mean(tensor, dim=1).squeeze(0).cpu().numpy())
                viz_data[name] = {"shape": list(arr.shape), "values": arr.tolist()}

        spec_np = np.nan_to_num(spectrogram.squeeze(0).squeeze(0).cpu().numpy())

        step = max(1, len(audio_data) // MAX_WAVEFORM_POINTS)
        waveform_data = audio_data[::step] if len(audio_data) > MAX_WAVEFORM_POINTS else audio_data

        return {
            "predictions": predictions,
            "visualization": viz_data,
            "input_spectrogram": {"shape": list(spec_np.shape), "values": spec_np.tolist()},
            "waveform": {
                "values": waveform_data.tolist(),
                "sample_rate": TARGET_SR,
                "duration": len(audio_data) / TARGET_SR,
            },
        }


def create_app(model_path: str) -> FastAPI:
    classifier = AudioClassifier(model_path)
    app = FastAPI(title="Audio CNN (local)")
    # The dashboard runs on localhost:3000, so the browser needs CORS to call :8000.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    def health():
        return {"status": "ok", "device": str(classifier.device), "classes": len(classifier.classes)}

    @app.post("/inference")
    def inference(request: InferenceRequest):
        try:
            return classifier.predict(request.audio_data)
        except Exception as e:  # bad/unsupported audio -> 400 instead of a 500 stack trace
            raise HTTPException(status_code=400, detail=f"Could not process audio: {e}")

    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Audio CNN inference server locally.")
    parser.add_argument("--model", default="best_model.pth", help="Path to the trained checkpoint")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    uvicorn.run(create_app(args.model), host=args.host, port=args.port)
