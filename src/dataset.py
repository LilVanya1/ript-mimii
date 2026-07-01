"""Dataset loading, windowing, feature extraction, and augmentation."""

import hashlib
import numpy as np
import librosa
import soundfile as sf
import torch
from torch.utils.data import Dataset
from pathlib import Path
import logging

from src.config import (
    DATA_DIR, SAMPLE_RATE, WINDOW_SEC, HOP_SEC,
    N_FFT, HOP_LENGTH, N_MELS, N_MFCC, RANDOM_SEED,
)

logger = logging.getLogger(__name__)
_DATASET_CACHE: dict[tuple, tuple] = {}


def _dataset_cache_key(machine_type: str, snr_db: int, machine_id: str | None,
                       val_ratio: float, test_ratio: float, seed: int,
                       augment_train: bool, norm_stats: tuple[float, float] | None) -> tuple:
    """Stable cache key for build_datasets results."""
    norm_key = None
    if norm_stats is not None:
        norm_key = (round(float(norm_stats[0]), 5), round(float(norm_stats[1]), 5))
    base = f"{machine_type}|{snr_db}|{machine_id}|{val_ratio:.4f}|{test_ratio:.4f}|{seed}|{augment_train}|{norm_key}"
    return (hashlib.sha1(base.encode("utf-8")).hexdigest(),)


def discover_files(machine_type: str, snr_db: int = 6, machine_id: str | None = None):
    # New layout (preferred): data/{snr}_dB_{machine_type}/{machine_type}/id_xx/...
    # Legacy layout (fallback): data/{machine_type}/id_xx/...
    base_candidates = [
        DATA_DIR / f"{snr_db}_dB_{machine_type}" / machine_type,
        DATA_DIR / machine_type,
    ]
    base = next((p for p in base_candidates if p.exists()), None)
    result = {"normal": {}, "abnormal": {}}

    if base is None:
        raise FileNotFoundError(
            f"Data not found for {machine_type} @ {snr_db}dB. "
            f"Expected one of: {[str(p) for p in base_candidates]}"
        )

    for mid_dir in sorted(base.iterdir()):
        if not mid_dir.is_dir():
            continue
        mid = mid_dir.name
        if machine_id and mid != machine_id:
            continue
        for label in ("normal", "abnormal"):
            label_dir = mid_dir / label
            if label_dir.exists():
                files = sorted(label_dir.glob("*.wav"))
                if files:
                    result[label].setdefault(mid, []).extend(files)

    return result


def load_audio(path: Path, sr: int = SAMPLE_RATE) -> np.ndarray:
    y, file_sr = sf.read(str(path), dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    if file_sr != sr:
        y = librosa.resample(y, orig_sr=file_sr, target_sr=sr)
    return y.astype(np.float32, copy=False)


def window_signal(y: np.ndarray, sr: int = SAMPLE_RATE,
                   window_sec: float = WINDOW_SEC,
                   hop_sec: float = HOP_SEC) -> list[np.ndarray]:
    win_len = int(window_sec * sr)
    hop_len = int(hop_sec * sr)
    windows = []
    start = 0
    while start + win_len <= len(y):
        windows.append(y[start:start + win_len])
        start += hop_len
    return windows


def extract_mel_spectrogram(y: np.ndarray, sr: int = SAMPLE_RATE,
                             norm_stats: tuple[float, float] | None = None) -> np.ndarray:
    """Log-mel spectrogram in absolute dB (ref=1.0, NOT per-window peak).

    Per-window `ref=np.max` + per-window min-max normalization destroys the
    absolute loudness/dynamic-range information that is often exactly the
    anomaly signal (e.g. louder impulses/clanks vs. steady hum). Instead we
    compute an absolute-reference dB spectrogram and normalize with GLOBAL
    stats (`norm_stats`) fit once on normal training data, so every window
    (train/val/test/inference, normal or abnormal) is mapped consistently.
    """
    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS,
    )
    log_S = librosa.power_to_db(S, ref=1.0, amin=1e-10)
    if norm_stats is None:
        return log_S.astype(np.float32, copy=False)
    min_v, max_v = norm_stats
    if max_v - min_v < 1e-8:
        return np.zeros_like(log_S, dtype=np.float32)
    norm = np.clip((log_S - min_v) / (max_v - min_v), 0.0, 1.0)
    return norm.astype(np.float32, copy=False)


def summarize_db_separation(normal_windows: list[np.ndarray],
                            abnormal_windows: list[np.ndarray],
                            sr: int = SAMPLE_RATE,
                            sample_size: int = 40) -> dict:
    """Compare raw dB ranges before global norm_stats are applied."""

    def _sample_stats(windows: list[np.ndarray]) -> tuple[float, float]:
        if not windows:
            return 0.0, 0.0
        rng = np.random.default_rng(RANDOM_SEED)
        idx = rng.choice(len(windows), size=min(sample_size, len(windows)), replace=False)
        mins, maxs = [], []
        for i in idx:
            log_S = extract_mel_spectrogram(windows[i], sr, norm_stats=None)
            mins.append(float(np.percentile(log_S, 1)))
            maxs.append(float(np.percentile(log_S, 99)))
        return float(np.mean(mins)), float(np.mean(maxs))

    n_min, n_max = _sample_stats(normal_windows)
    a_min, a_max = _sample_stats(abnormal_windows)
    return {
        "normal_db_min": n_min,
        "normal_db_max": n_max,
        "abnormal_db_min": a_min,
        "abnormal_db_max": a_max,
        "db_min_delta": a_min - n_min,
        "db_max_delta": a_max - n_max,
    }


def compute_norm_stats(windows: list[np.ndarray], sr: int = SAMPLE_RATE,
                       sample_size: int = 300) -> tuple[float, float]:
    """Fit global [min, max] dB stats from a sample of NORMAL training windows.

    Uses the 1st/99th percentile (not raw min/max) so a single outlier
    window can't blow up the range for everyone else.
    """
    rng = np.random.default_rng(RANDOM_SEED)
    n = len(windows)
    idx = rng.choice(n, size=min(sample_size, n), replace=False)
    mins, maxs = [], []
    for i in idx:
        log_S = extract_mel_spectrogram(windows[i], sr, norm_stats=None)
        mins.append(float(np.percentile(log_S, 1)))
        maxs.append(float(np.percentile(log_S, 99)))
    return float(np.mean(mins)), float(np.mean(maxs))


def extract_mfcc(y: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    mfcc = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH,
    )
    return mfcc


def add_noise(y: np.ndarray, noise_level: float = 0.005) -> np.ndarray:
    rng = np.random.default_rng()
    noise = rng.normal(0, noise_level, y.shape)
    return y + noise


def change_volume(y: np.ndarray, gain_range: tuple = (0.7, 1.3)) -> np.ndarray:
    rng = np.random.default_rng()
    gain = rng.uniform(*gain_range)
    return y * gain


def augment_waveform(y: np.ndarray) -> np.ndarray:
    """Fast augmentations only (no time_stretch)."""
    rng = np.random.default_rng()
    out = y.copy()
    if rng.random() < 0.5:
        out = add_noise(out)
    if rng.random() < 0.5:
        out = change_volume(out)
    return out


def spec_augment(mel: torch.Tensor, freq_mask_param: int = 8, time_mask_param: int = 16) -> torch.Tensor:
    """SpecAugment on mel tensor (1, n_mels, T) — runs on GPU."""
    rng = torch.rand(2)
    if rng[0] < 0.5:
        f = torch.randint(1, freq_mask_param + 1, (1,)).item()
        f0 = torch.randint(0, mel.shape[1] - f + 1, (1,)).item()
        mel = mel.clone()
        mel[:, f0:f0 + f, :] = 0.0
    if rng[1] < 0.5:
        t = torch.randint(1, time_mask_param + 1, (1,)).item()
        t0 = torch.randint(0, mel.shape[2] - t + 1, (1,)).item()
        mel = mel.clone()
        mel[:, :, t0:t0 + t] = 0.0
    return mel


class PrecomputedMelDataset(Dataset):
    """Precomputes ALL mel spectrograms once in __init__ → zero CPU overhead during training."""

    def __init__(self, windows: list[np.ndarray], labels: np.ndarray | None = None,
                 augment_normal: bool = False, sr: int = SAMPLE_RATE,
                 progress_cb=None, norm_stats: tuple[float, float] | None = None):
        self.labels = labels
        self.augment_normal = augment_normal
        n = len(windows)
        logger.info(f"Precomputing {n} mel spectrograms...")

        # Compute all spectrograms upfront
        specs = []
        for i, y in enumerate(windows):
            if augment_normal and (labels is None or labels[i] == 0):
                y = augment_waveform(y)
            mel = extract_mel_spectrogram(y, sr, norm_stats=norm_stats)
            specs.append(mel)
            if progress_cb and i % 500 == 0 and i > 0:
                progress_cb(i, n)

        # Stack into single tensor (N, 1, n_mels, T)
        arr = np.stack(specs, axis=0)
        self.data = torch.from_numpy(arr).float().unsqueeze(1)  # (N, 1, n_mels, T)

        if labels is not None:
            self.label_tensor = torch.from_numpy(labels).long()
        else:
            self.label_tensor = None

        logger.info(f"Precomputed tensor shape: {list(self.data.shape)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        if self.augment_normal:
            x = spec_augment(x)
        if self.label_tensor is not None:
            return x, int(self.label_tensor[idx])
        return x


# Keep old name as alias for compatibility
MelSpectrogramDataset = PrecomputedMelDataset


def _windows_from_files(paths: list[Path], progress_cb=None, processed_ref=None, total_files=None):
    windows = []
    for p in paths:
        y = load_audio(p)
        windows.extend(window_signal(y))
        if processed_ref is not None:
            processed_ref[0] += 1
            if progress_cb and total_files:
                done = processed_ref[0]
                if done % 25 == 0 or done == total_files:
                    progress_cb(done, total_files)
    return windows


def build_datasets(machine_type: str, snr_db: int = 6,
                   val_ratio: float = 0.15, test_ratio: float = 0.15,
                   augment_train: bool = True, seed: int = RANDOM_SEED,
                   progress_cb=None, machine_id: str | None = None,
                   norm_stats: tuple[float, float] | None = None):
    cache_key = _dataset_cache_key(
        machine_type=machine_type,
        snr_db=snr_db,
        machine_id=machine_id,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
        augment_train=augment_train,
        norm_stats=norm_stats,
    )
    if cache_key in _DATASET_CACHE:
        logger.info(f"[{machine_type} @ {snr_db}dB] dataset cache hit")
        train_ds, val_ds, test_ds, test_labels, test_abnormal_ids, stats = _DATASET_CACHE[cache_key]
        return train_ds, val_ds, test_ds, test_labels.copy(), list(test_abnormal_ids), stats

    rng = np.random.default_rng(seed)
    files_info = discover_files(machine_type, snr_db, machine_id=machine_id)

    normal_paths = []
    abnormal_windows = []
    abnormal_ids = []
    n_normal_files = sum(len(v) for v in files_info["normal"].values())
    n_abnormal_files = sum(len(v) for v in files_info["abnormal"].values())
    total_files = n_normal_files + n_abnormal_files
    processed_files = [0]

    for mid, paths in files_info["normal"].items():
        normal_paths.extend(paths)

    # Split by FILES first (avoid leakage of windows from same file).
    if not normal_paths:
        raise FileNotFoundError(
            f"No normal audio files found for {machine_type} @ {snr_db}dB"
        )

    normal_paths = list(normal_paths)
    n_files = len(normal_paths)
    file_idx = rng.permutation(n_files)
    n_test_files = max(1, int(n_files * test_ratio))
    n_val_files = max(1, int(n_files * val_ratio))
    n_train_files = n_files - n_test_files - n_val_files

    # Keep all three splits non-empty for stable threshold/eval.
    while n_train_files < 1 and n_test_files > 1:
        n_test_files -= 1
        n_train_files = n_files - n_test_files - n_val_files
    while n_train_files < 1 and n_val_files > 1:
        n_val_files -= 1
        n_train_files = n_files - n_test_files - n_val_files
    n_train_files = max(1, n_train_files)

    test_file_idx = file_idx[:n_test_files]
    val_file_idx = file_idx[n_test_files:n_test_files + n_val_files]
    train_file_idx = file_idx[n_test_files + n_val_files:n_test_files + n_val_files + n_train_files]

    train_files = [normal_paths[i] for i in train_file_idx]
    val_files = [normal_paths[i] for i in val_file_idx]
    test_normal_files = [normal_paths[i] for i in test_file_idx]

    train_windows = _windows_from_files(train_files, progress_cb=progress_cb, processed_ref=processed_files, total_files=total_files)
    val_normal_windows = _windows_from_files(val_files, progress_cb=progress_cb, processed_ref=processed_files, total_files=total_files)
    test_normal_windows = _windows_from_files(test_normal_files, progress_cb=progress_cb, processed_ref=processed_files, total_files=total_files)

    for mid, paths in files_info["abnormal"].items():
        for p in paths:
            y = load_audio(p)
            wins = window_signal(y)
            abnormal_windows.extend(wins)
            abnormal_ids.extend([mid] * len(wins))
            processed_files[0] += 1
            if progress_cb and (processed_files[0] % 25 == 0 or processed_files[0] == total_files):
                progress_cb(processed_files[0], total_files)

    test_windows = test_normal_windows + abnormal_windows
    test_labels = np.array(
        [0] * len(test_normal_windows) + [1] * len(abnormal_windows)
    )
    test_abnormal_ids = [None] * len(test_normal_windows) + abnormal_ids

    logger.info(f"[{machine_type} @ {snr_db}dB] train={len(train_windows)} "
                f"val={len(val_normal_windows)} test_normal={len(test_normal_windows)} "
                f"test_abnormal={len(abnormal_windows)}")

    # Fit global dB normalization stats ONCE on normal training windows only,
    # then reuse identically for val/test (normal AND abnormal). Per-window
    # min-max normalization was collapsing absolute loudness differences,
    # which is often the actual anomaly signal.
    if norm_stats is None:
        norm_stats = compute_norm_stats(train_windows, sr=SAMPLE_RATE)
    logger.info(f"[{machine_type} @ {snr_db}dB] norm_stats(min,max)={norm_stats}")

    db_sep = summarize_db_separation(train_windows, abnormal_windows, sr=SAMPLE_RATE)
    logger.info(
        f"[{machine_type} @ {snr_db}dB] raw dB separation | "
        f"normal=({db_sep['normal_db_min']:.2f},{db_sep['normal_db_max']:.2f}) "
        f"abnormal=({db_sep['abnormal_db_min']:.2f},{db_sep['abnormal_db_max']:.2f}) "
        f"delta_min={db_sep['db_min_delta']:.2f} delta_max={db_sep['db_max_delta']:.2f}"
    )

    train_ds = PrecomputedMelDataset(train_windows, augment_normal=augment_train,
                                     progress_cb=progress_cb, norm_stats=norm_stats)
    val_ds = PrecomputedMelDataset(val_normal_windows, norm_stats=norm_stats)
    test_ds = PrecomputedMelDataset(test_windows, labels=test_labels, norm_stats=norm_stats)

    print(f"[{machine_type} @ {snr_db}dB] train={len(train_windows)}  "
          f"val={len(val_normal_windows)}  test_normal={len(test_normal_windows)}  "
          f"test_abnormal={len(abnormal_windows)}")

    _DATASET_CACHE[cache_key] = (
        train_ds, val_ds, test_ds, test_labels.copy(), list(test_abnormal_ids), norm_stats
    )
    return train_ds, val_ds, test_ds, test_labels, test_abnormal_ids, norm_stats
