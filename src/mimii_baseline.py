"""Port of official MIMII dense AE baseline (Hitachi) to PyTorch."""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

from src.config import RANDOM_SEED
from src.dataset import discover_files

logger = logging.getLogger(__name__)

BASELINE_N_MELS = 64
BASELINE_FRAMES = 5
BASELINE_N_FFT = 1024
BASELINE_HOP_LENGTH = 512
BASELINE_POWER = 2.0
BASELINE_LATENT = 8
BASELINE_HIDDEN = 64
BASELINE_EPOCHS = 50
BASELINE_BATCH_SIZE = 512
BASELINE_LR = 1e-3
BASELINE_PATIENCE = 8
BASELINE_VAL_SPLIT = 0.1


@dataclass(frozen=True)
class BaselineFeatureConfig:
    n_mels: int = BASELINE_N_MELS
    frames: int = BASELINE_FRAMES
    n_fft: int = BASELINE_N_FFT
    hop_length: int = BASELINE_HOP_LENGTH
    power: float = BASELINE_POWER

    @property
    def input_dim(self) -> int:
        return self.n_mels * self.frames

    def to_dict(self) -> dict:
        return {
            "backend": "mimii_baseline",
            "n_mels": self.n_mels,
            "frames": self.frames,
            "n_fft": self.n_fft,
            "hop_length": self.hop_length,
            "power": self.power,
            "input_dim": self.input_dim,
        }

    @classmethod
    def from_dict(cls, data: dict | None) -> BaselineFeatureConfig:
        if not data:
            return cls()
        return cls(
            n_mels=int(data.get("n_mels", BASELINE_N_MELS)),
            frames=int(data.get("frames", BASELINE_FRAMES)),
            n_fft=int(data.get("n_fft", BASELINE_N_FFT)),
            hop_length=int(data.get("hop_length", BASELINE_HOP_LENGTH)),
            power=float(data.get("power", BASELINE_POWER)),
        )


def _load_audio_native_sr(path: Path) -> tuple[int, np.ndarray]:
    y, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    return int(sr), y.astype(np.float32, copy=False)


def file_to_vector_array(path: Path, cfg: BaselineFeatureConfig) -> np.ndarray:
    """Official baseline feature: concat `frames` log-mel columns into one vector."""
    sr, y = _load_audio_native_sr(path)
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        n_mels=cfg.n_mels,
        power=cfg.power,
    )
    log_mel = (20.0 / cfg.power) * np.log10(mel + sys.float_info.epsilon)
    dims = cfg.input_dim
    n_frames = log_mel.shape[1]
    vector_size = n_frames - cfg.frames + 1
    if vector_size < 1:
        return np.empty((0, dims), dtype=np.float32)

    out = np.zeros((vector_size, dims), dtype=np.float32)
    for t in range(cfg.frames):
        out[:, cfg.n_mels * t: cfg.n_mels * (t + 1)] = log_mel[:, t: t + vector_size].T
    return out


def split_baseline_files(
    normal_paths: list[Path],
    abnormal_paths: list[Path],
) -> tuple[list[Path], list[Path], np.ndarray]:
    """Official split: train=remaining normal, eval=matched normal + all abnormal."""
    normal_paths = sorted(normal_paths)
    abnormal_paths = sorted(abnormal_paths)
    if not normal_paths:
        raise FileNotFoundError("No normal wav files for baseline split")
    if not abnormal_paths:
        raise FileNotFoundError("No abnormal wav files for baseline split")

    n_abn = len(abnormal_paths)
    if len(normal_paths) <= n_abn:
        raise ValueError(
            f"Need more normal files than abnormal for MIMII baseline split "
            f"(normal={len(normal_paths)}, abnormal={n_abn})"
        )

    train_files = normal_paths[n_abn:]
    eval_files = normal_paths[:n_abn] + abnormal_paths
    eval_labels = np.array([0] * n_abn + [1] * n_abn, dtype=np.int64)
    return train_files, eval_files, eval_labels


def vectors_from_files(
    paths: list[Path],
    cfg: BaselineFeatureConfig,
    progress_cb=None,
) -> np.ndarray:
    chunks: list[np.ndarray] = []
    total = len(paths)
    for idx, path in enumerate(paths):
        arr = file_to_vector_array(path, cfg)
        if arr.size:
            chunks.append(arr)
        if progress_cb and total:
            progress_cb(idx + 1, total)
    if not chunks:
        return np.empty((0, cfg.input_dim), dtype=np.float32)
    return np.vstack(chunks)


class MimiiBaselineAE(nn.Module):
    """Dense AE 64-64-8-64-64 from official MIMII baseline."""

    def __init__(self, input_dim: int = BASELINE_N_MELS * BASELINE_FRAMES):
        super().__init__()
        self.input_dim = int(input_dim)
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, BASELINE_HIDDEN),
            nn.ReLU(),
            nn.Linear(BASELINE_HIDDEN, BASELINE_HIDDEN),
            nn.ReLU(),
            nn.Linear(BASELINE_HIDDEN, BASELINE_LATENT),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(BASELINE_LATENT, BASELINE_HIDDEN),
            nn.ReLU(),
            nn.Linear(BASELINE_HIDDEN, BASELINE_HIDDEN),
            nn.ReLU(),
            nn.Linear(BASELINE_HIDDEN, self.input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))

    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        recon = self.forward(x)
        return torch.mean((x - recon) ** 2, dim=1)


def _split_train_val_vectors(
    train_vectors: np.ndarray,
    val_split: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    n = len(train_vectors)
    if n < 2:
        raise ValueError("Not enough baseline training vectors")
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_val = max(1, int(n * val_split))
    n_val = min(n_val, n - 1)
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]
    return train_vectors[train_idx], train_vectors[val_idx]


def train_baseline_ae(
    model: MimiiBaselineAE,
    train_vectors: np.ndarray,
    epochs: int,
    lr: float,
    patience: int,
    device: torch.device,
    save_path: Path | None = None,
    cb=None,
    val_split: float = BASELINE_VAL_SPLIT,
    batch_size: int = BASELINE_BATCH_SIZE,
    seed: int = RANDOM_SEED,
) -> tuple[MimiiBaselineAE, dict]:
    train_x, val_x = _split_train_val_vectors(train_vectors, val_split, seed)
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(train_x).float()),
        batch_size=min(batch_size, len(train_x)),
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(val_x).float()),
        batch_size=min(batch_size, len(val_x)),
        shuffle=False,
    )

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_val = float("inf")
    wait = 0
    history = {"train_loss": [], "val_loss": [], "lr": []}

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for (batch,) in train_loader:
            x = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), x)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))

        model.eval()
        val_losses = []
        with torch.no_grad():
            for (batch,) in val_loader:
                x = batch.to(device)
                val_losses.append(float(criterion(model(x), x).item()))

        t_loss = float(np.mean(train_losses))
        v_loss = float(np.mean(val_losses))
        history["train_loss"].append(t_loss)
        history["val_loss"].append(v_loss)
        history["lr"].append(lr)

        if cb:
            cb(epoch, t_loss, v_loss)
        else:
            print(f"[baseline] Epoch {epoch:3d} | train={t_loss:.6f} | val={v_loss:.6f}")

        if v_loss < best_val:
            best_val = v_loss
            wait = 0
            if save_path:
                save_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "backend": "mimii_baseline",
                        "state_dict": model.state_dict(),
                        "feature": BaselineFeatureConfig().to_dict(),
                    },
                    save_path,
                )
        else:
            wait += 1
            if wait >= patience:
                print(f"[baseline] Early stopping at epoch {epoch}")
                break

    if save_path and save_path.exists():
        raw = torch.load(save_path, map_location=device, weights_only=True)
        model.load_state_dict(raw["state_dict"] if isinstance(raw, dict) else raw)

    return model, history


def compute_baseline_threshold(
    model: MimiiBaselineAE,
    val_vectors: np.ndarray,
    device: torch.device,
    quantile: float = 0.90,
) -> float:
    loader = DataLoader(
        TensorDataset(torch.from_numpy(val_vectors).float()),
        batch_size=512,
        shuffle=False,
    )
    model.eval()
    scores = []
    with torch.no_grad():
        for (batch,) in loader:
            scores.extend(model.reconstruction_error(batch.to(device)).cpu().numpy())
    scores = np.asarray(scores, dtype=np.float64)
    threshold = float(np.quantile(scores, quantile)) if scores.size else 0.0
    print(f"[baseline] threshold q={quantile:.3f}: {threshold:.6f}")
    return threshold


def file_level_scores(
    model: MimiiBaselineAE,
    eval_files: list[Path],
    cfg: BaselineFeatureConfig,
    device: torch.device,
) -> np.ndarray:
    """Official eval: one score per wav = mean vector MSE inside file."""
    model.eval()
    scores = []
    with torch.no_grad():
        for path in eval_files:
            vectors = file_to_vector_array(path, cfg)
            if vectors.size == 0:
                scores.append(0.0)
                continue
            x = torch.from_numpy(vectors).float().to(device)
            err = model.reconstruction_error(x).cpu().numpy()
            scores.append(float(np.mean(err)))
    return np.asarray(scores, dtype=np.float64)


def evaluate_baseline_file_level(
    model: MimiiBaselineAE,
    eval_files: list[Path],
    eval_labels: np.ndarray,
    cfg: BaselineFeatureConfig,
    device: torch.device,
    threshold: float | None = None,
) -> dict:
    scores = file_level_scores(model, eval_files, cfg, device)
    auc = float(roc_auc_score(eval_labels, scores))
    out = {"auc_roc": auc, "errors": scores, "eval_labels": eval_labels}
    if threshold is not None:
        preds = (scores > threshold).astype(int)
        out["preds"] = preds
    print(f"[baseline] file-level AUC: {auc:.4f}")
    return out


def build_baseline_pack(
    machine_type: str,
    snr_db: int = 6,
    machine_id: str | None = None,
    cfg: BaselineFeatureConfig | None = None,
    progress_cb=None,
    seed: int = RANDOM_SEED,
) -> dict:
    cfg = cfg or BaselineFeatureConfig()
    files_info = discover_files(machine_type, snr_db, machine_id=machine_id)

    normal_paths: list[Path] = []
    abnormal_paths: list[Path] = []
    for paths in files_info["normal"].values():
        normal_paths.extend(paths)
    for paths in files_info["abnormal"].values():
        abnormal_paths.extend(paths)

    train_files, eval_files, eval_labels = split_baseline_files(normal_paths, abnormal_paths)
    train_vectors = vectors_from_files(train_files, cfg, progress_cb=progress_cb)
    _, val_vectors = _split_train_val_vectors(train_vectors, BASELINE_VAL_SPLIT, seed)

    logger.info(
        f"[baseline {machine_type}@{snr_db}dB {machine_id}] "
        f"train_files={len(train_files)} train_vectors={len(train_vectors)} "
        f"eval_files={len(eval_files)}"
    )
    return {
        "cfg": cfg,
        "train_files": train_files,
        "train_vectors": train_vectors,
        "val_vectors": val_vectors,
        "eval_files": eval_files,
        "eval_labels": eval_labels,
    }


def save_baseline_meta(path: Path, cfg: BaselineFeatureConfig) -> None:
    import json

    with path.open("w", encoding="utf-8") as f:
        json.dump(cfg.to_dict(), f)


def load_baseline_meta(path: Path) -> BaselineFeatureConfig | None:
    import json

    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return BaselineFeatureConfig.from_dict(json.load(f))
    except Exception:
        return None
