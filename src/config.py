from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
MODEL_DIR = ROOT_DIR / "models"
RESULTS_DIR = ROOT_DIR / "results"

SAMPLE_RATE = 16000
WINDOW_SEC = 3.0
HOP_SEC = 1.0  # 50% overlap
N_FFT = 2048
HOP_LENGTH = 256
N_MELS = 128
N_MFCC = 13

MACHINE_TYPES = ["fan", "pump", "slider", "valve"]

# SNR levels available in MIMII
SNR_LEVELS = [-6, 0, 6]

# MobileNetV2 one-class speed profile
LATENT_DIM = 8
LATENT_L1 = 7e-4
BATCH_SIZE = 96
BASE_CHANNELS = 24
LEARNING_RATE = 3e-4
EPOCHS = 90
PATIENCE = 12

ANOMALY_QUANTILE = 0.90
THRESHOLD_METHOD = "mad"   # kde_fpr | mad | quantile
THRESHOLD_TARGET_FPR = 0.05
THRESHOLD_MAD_K = 2.2

RANDOM_SEED = 42

# Official MIMII baseline (Hitachi) defaults — see src/mimii_baseline.py
BASELINE_EPOCHS = 50
BASELINE_BATCH_SIZE = 512
BASELINE_LR = 1e-3

MIMII_ZENODO_IDS = {
    "fan":    "3384388",
    "pump":   "3384388",
    "slider": "3384388",
    "valve":  "3384388",
}

# Baseline Dense Autoencoder (MIMII paper reproduction)
BASELINE_N_MELS = 64
BASELINE_FRAMES = 5
BASELINE_N_FFT = 1024
BASELINE_HOP_LENGTH = 512
BASELINE_POWER = 2.0
