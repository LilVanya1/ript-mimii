"""Acoustic diagnostics — ConvAutoencoder on CUDA. Full GUI."""

import os, sys, threading, traceback, logging, tempfile, argparse, json, shutil
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")

from flask import Flask, jsonify, render_template, request, send_file
from werkzeug.utils import secure_filename

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(ROOT, "app.log"), encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", os.urandom(24).hex())
PORT = 228
_state_lock = threading.Lock()

from src.config import (
    MODEL_DIR, DATA_DIR, MACHINE_TYPES, RESULTS_DIR,
    BATCH_SIZE, EPOCHS, LEARNING_RATE, PATIENCE, ANOMALY_QUANTILE,
    THRESHOLD_METHOD, THRESHOLD_TARGET_FPR, THRESHOLD_MAD_K, LATENT_DIM, LATENT_L1, BASE_CHANNELS,
    BASELINE_N_MELS, BASELINE_FRAMES, BASELINE_N_FFT, BASELINE_HOP_LENGTH, BASELINE_POWER,
)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_HISTORY_DIR = MODEL_DIR / "history"
MODEL_HISTORY_DIR.mkdir(parents=True, exist_ok=True)
MODEL_REGISTRY_PATH = MODEL_DIR / "model_registry.json"

state = {
    "status":           "idle",
    "progress":         0,
    "log":              [],
    "startup_log":      [],
    "results":          {},
    "available_models": [],
    "model_registry":   {},
    "train_history":    [],
    "train_config":     {
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "patience": PATIENCE,
        "anomaly_quantile": ANOMALY_QUANTILE,
        "threshold_method": THRESHOLD_METHOD,
        "threshold_target_fpr": THRESHOLD_TARGET_FPR,
        "threshold_mad_k": THRESHOLD_MAD_K,
        "latent_dim": LATENT_DIM,
        "latent_l1": LATENT_L1,
        "base_channels": BASE_CHANNELS,
    },
    "tune_config":      {
        "trials": 12,
        "trial_epochs": 20,
        "trial_patience": 5,
        "threshold_method": THRESHOLD_METHOD,
        "threshold_target_fpr": THRESHOLD_TARGET_FPR,
        "threshold_mad_k": THRESHOLD_MAD_K,
        "anomaly_quantile": ANOMALY_QUANTILE,
    },
    "device":           "cpu",
}
_model_cache: dict = {}  # machine_type -> {"model": ConvAutoencoder, "threshold": float}
_stop_event = threading.Event()
MAX_LOG = 500


def _update_state(**kwargs):
    with _state_lock:
        for k, v in kwargs.items():
            if k in state:
                state[k] = v


def log(msg: str, startup: bool = False):
    dst = state["startup_log"] if startup else state["log"]
    dst.append(msg)
    if len(dst) > MAX_LOG:
        del dst[:len(dst) - MAX_LOG]
    logger.info(msg)


def _model_path(mt: str) -> Path:
    return MODEL_DIR / f"conv_ae_{mt}.pt"


def _threshold_path(mt: str) -> Path:
    return MODEL_DIR / f"conv_ae_{mt}_threshold.npy"


def _model_key(machine_type: str, machine_id: str | None, snr_db: int | None = None) -> str:
    base = f"{machine_type}_{machine_id}" if machine_id else machine_type
    return f"{base}_{int(snr_db)}db" if snr_db is not None else base


def _normalize_machine_id(raw_value: str | None, default_id: str = "id_00") -> str:
    """Normalize UI machine_id into strict per-ID mode."""
    mid = (raw_value or "").strip()
    if not mid:
        return default_id
    lower_mid = mid.lower()
    if lower_mid in {"all", "all id", "all ids"}:
        return default_id
    if "все" in lower_mid:
        return default_id
    return mid


def _model_path_for(machine_type: str, machine_id: str | None, snr_db: int) -> Path:
    return MODEL_DIR / f"conv_ae_{_model_key(machine_type, machine_id, snr_db)}.pt"


def _threshold_path_for(machine_type: str, machine_id: str | None, snr_db: int) -> Path:
    return MODEL_DIR / f"conv_ae_{_model_key(machine_type, machine_id, snr_db)}_threshold.npy"


def _norm_stats_path_for(machine_type: str, machine_id: str | None, snr_db: int) -> Path:
    return MODEL_DIR / f"conv_ae_{_model_key(machine_type, machine_id, snr_db)}_norm.json"


def _save_norm_stats(path: Path, stats: tuple[float, float]):
    with path.open("w", encoding="utf-8") as f:
        json.dump({"min": stats[0], "max": stats[1]}, f)


def _load_norm_stats(path: Path) -> tuple[float, float] | None:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            d = json.load(f)
        return float(d["min"]), float(d["max"])
    except Exception:
        return None





def _extract_state_dict(raw_obj):
    if isinstance(raw_obj, dict) and "state_dict" in raw_obj and isinstance(raw_obj["state_dict"], dict):
        return raw_obj["state_dict"]
    return raw_obj


def _checkpoint_compatibility(model, state_dict: dict):
    model_sd = model.state_dict()
    missing = []
    mismatched = []
    unexpected = []

    for k, v in state_dict.items():
        if k not in model_sd:
            unexpected.append(k)
            continue
        if tuple(v.shape) != tuple(model_sd[k].shape):
            mismatched.append((k, tuple(v.shape), tuple(model_sd[k].shape)))

    for k in model_sd.keys():
        if k not in state_dict:
            missing.append(k)

    if missing or mismatched:
        parts = []
        if missing:
            parts.append(f"missing={len(missing)}")
        if mismatched:
            ex = ", ".join([f"{k}:{old}->{new}" for k, old, new in mismatched[:2]])
            parts.append(f"shape_mismatch={len(mismatched)} ({ex})")
        if unexpected:
            parts.append(f"unexpected={len(unexpected)}")
        return False, "; ".join(parts)
    return True, "ok"


def _load_model_weights_checked(model, path: Path, device):
    import torch
    raw = torch.load(path, map_location=device, weights_only=True)
    state_dict = _extract_state_dict(raw)
    ok, reason = _checkpoint_compatibility(model, state_dict)
    if not ok:
        raise ValueError(reason)
    model.load_state_dict(state_dict)
    return model


def _load_registry() -> dict:
    if not MODEL_REGISTRY_PATH.exists():
        return {"models": []}
    try:
        with MODEL_REGISTRY_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and isinstance(data.get("models"), list):
            return data
    except Exception as e:
        log(f"WARNING registry load failed: {e}", startup=True)
    return {"models": []}


def _save_registry(registry: dict):
    tmp = MODEL_REGISTRY_PATH.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(registry, f, ensure_ascii=False, indent=2)
    tmp.replace(MODEL_REGISTRY_PATH)


def _refresh_registry_state(registry: dict):
    grouped = {}
    for rec in registry.get("models", []):
        grouped.setdefault(rec["machine_type"], []).append(rec)
    for mt in grouped:
        grouped[mt].sort(key=lambda x: x.get("created_at", ""), reverse=True)
    state["model_registry"] = grouped
    all_items = sorted(registry.get("models", []), key=lambda x: x.get("created_at", ""), reverse=True)
    state["train_history"] = all_items[:30]


def _ensure_registry_compat():
    """Backfill optional fields for old registry entries."""
    reg = _load_registry()
    changed = False
    for m in reg.get("models", []):
        if "machine_id" not in m:
            m["machine_id"] = None
            changed = True
        if "norm_stats_path" not in m:
            m["norm_stats_path"] = None
            changed = True
    if changed:
        _save_registry(reg)
    _refresh_registry_state(reg)


def _register_model(machine_type: str, machine_id: str | None, snr_db: int, threshold: float, metrics: dict,
                    mode: str, base_model_id: str | None, epochs_trained: int,
                    model_path: Path, threshold_path: Path, norm_stats_path: Path):
    registry = _load_registry()
    model_id = f"{_model_key(machine_type, machine_id, snr_db)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    rec = {
        "id": model_id,
        "machine_type": machine_type,
        "machine_id": machine_id,
        "snr": snr_db,
        "mode": mode,
        "base_model_id": base_model_id,
        "epochs_trained": epochs_trained,
        "threshold": round(float(threshold), 6),
        "auc_roc": round(float(metrics.get("auc_roc", 0.0)), 4),
        "pauc": round(float(metrics.get("pauc", 0.0)), 4),
        "f1": round(float(metrics.get("f1", 0.0)), 4),
        "accuracy": round(float(metrics.get("accuracy", 0.0)), 4),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model_path": str(model_path.relative_to(ROOT)),
        "threshold_path": str(threshold_path.relative_to(ROOT)),
        "norm_stats_path": str(norm_stats_path.relative_to(ROOT)) if norm_stats_path.exists() else None,
    }
    registry["models"].append(rec)
    _save_registry(registry)
    _refresh_registry_state(registry)
    return rec


def _latest_model_record(machine_type: str, machine_id: str | None = None) -> dict | None:
    registry = _load_registry()
    items = [m for m in registry.get("models", []) if m.get("machine_type") == machine_type and (machine_id is None or m.get("machine_id") == machine_id)]
    if not items:
        return None
    items.sort(key=lambda x: x.get("created_at", ""), reverse=True)
    return items[0]


def _partial_auc(y_true, scores, max_fpr=0.1):
    from src.evaluate import partial_auc
    return partial_auc(y_true, scores, max_fpr)


def _build_tune_split(test_ds, test_labels, rng_seed: int = 42, keep_ratio: float = 0.4):
    """Build a stratified subset of test split for fast Optuna objective."""
    from torch.utils.data import Subset

    labels = np.asarray(test_labels)
    normal_idx = np.where(labels == 0)[0]
    abnormal_idx = np.where(labels == 1)[0]
    if normal_idx.size == 0 or abnormal_idx.size == 0:
        raise ValueError("Need both normal and abnormal windows for tuning objective.")

    rng = np.random.default_rng(rng_seed)
    n_norm = max(64, int(normal_idx.size * keep_ratio))
    n_abn = max(64, int(abnormal_idx.size * keep_ratio))
    n_norm = min(n_norm, normal_idx.size)
    n_abn = min(n_abn, abnormal_idx.size)

    pick_norm = rng.choice(normal_idx, size=n_norm, replace=False)
    pick_abn = rng.choice(abnormal_idx, size=n_abn, replace=False)
    tune_indices = np.concatenate([pick_norm, pick_abn])
    rng.shuffle(tune_indices)

    tune_ds = Subset(test_ds, tune_indices.tolist())
    tune_labels = labels[tune_indices].astype(np.int32, copy=False)
    return tune_ds, tune_labels


def _preload():
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state["device"] = "cuda" if torch.cuda.is_available() else "cpu"

    _ensure_registry_compat()

    from src.autoencoder import ConvAutoencoder
    # load all compatible current checkpoints from model dir
    for mp in sorted(MODEL_DIR.glob("conv_ae_*.pt")):
        if "_history" in mp.name:
            continue
        key = mp.stem.replace("conv_ae_", "")
        tp = MODEL_DIR / f"{mp.stem}_threshold.npy"
        if not tp.exists():
            continue
        try:
            model = ConvAutoencoder().to(device)
            _load_model_weights_checked(model, mp, device)
            model.eval()
            threshold = float(np.load(tp))
            norm_path = MODEL_DIR / f"{mp.stem}_norm.json"
            norm_stats = _load_norm_stats(norm_path)
            if norm_stats is None:
                log(f"WARNING {key}: missing norm stats ({norm_path.name}), "
                    f"scores will be inaccurate. Retrain to fix.", startup=True)
            _model_cache[key] = {"model": model, "threshold": threshold, "norm_stats": norm_stats}
            state["available_models"].append(key)
            log(f"Loaded: {mp.name}  thr={threshold:.6f}", startup=True)
        except Exception as e:
            log(f"WARNING {key}: incompatible checkpoint for current config ({e})", startup=True)

    if not state["available_models"]:
        log("No trained models found. Use Train tab.", startup=True)


# ── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/state")
def get_state():
    return jsonify({
        "status":           state["status"],
        "progress":         state["progress"],
        "log":              state["log"][-120:],
        "startup_log":      state["startup_log"],
        "results":          state["results"],
        "available_models": state["available_models"],
        "model_registry":   state["model_registry"],
        "train_history":    state["train_history"],
        "train_config":     state["train_config"],
        "tune_config":      state["tune_config"],
        "device":           state["device"],
    })


@app.route("/api/images")
def list_images():
    imgs = [f.name for f in sorted(RESULTS_DIR.glob("*.png"))] if RESULTS_DIR.exists() else []
    return jsonify(imgs)


@app.route("/results/<path:filename>")
def serve_result(filename):
    return send_file(RESULTS_DIR / filename)


@app.route("/api/download", methods=["POST"])
def download_data():
    if state["status"] == "running":
        return jsonify({"error": "Busy"}), 409
    machine_type = (request.json or {}).get("machine_type", "fan")

    def _run():
        _update_state(status="running", progress=0, log=[], results={})
        try:
            from src.download import download_mimii
            def progress_cb(downloaded, total):
                if total:
                    _update_state(progress=max(1, min(95, int(downloaded / total * 95))))
            log(f"Downloading MIMII: {machine_type}…")
            download_mimii([machine_type], progress_cb=progress_cb)
            _update_state(progress=100)
            log("Download complete!")
        except Exception as e:
            log(f"ERROR: {e}")
            traceback.print_exc()
        finally:
            _update_state(status="idle")

    threading.Thread(target=_run, daemon=True).start()
    return jsonify({"ok": True})


@app.route("/api/stop", methods=["POST"])
def stop_training():
    _stop_event.set()
    log("Stop requested by user")
    return jsonify({"ok": True})


@app.route("/api/train", methods=["POST"])
def train():
    if state["status"] == "running":
        return jsonify({"error": "Busy"}), 409
    data         = request.json or {}
    machine_type = data.get("machine_type", "fan")
    machine_id   = _normalize_machine_id(data.get("machine_id"))
    snr_db       = int(data.get("snr", 6))
    train_mode   = data.get("train_mode", "new")
    base_model_id = (data.get("base_model_id") or "").strip() or None
    train_epochs = int(data.get("epochs", EPOCHS))
    train_bs = int(data.get("batch_size", BATCH_SIZE))
    train_lr = float(data.get("learning_rate", LEARNING_RATE))
    train_patience = int(data.get("patience", PATIENCE))
    train_quantile = float(data.get("anomaly_quantile", ANOMALY_QUANTILE))
    train_threshold_method = (data.get("threshold_method", THRESHOLD_METHOD) or THRESHOLD_METHOD).strip().lower()
    train_target_fpr = float(data.get("threshold_target_fpr", THRESHOLD_TARGET_FPR))
    train_mad_k = float(data.get("threshold_mad_k", THRESHOLD_MAD_K))
    train_latent_l1 = float(data.get("latent_l1", LATENT_L1))
    train_latent_dim = int(data.get("latent_dim", state["train_config"].get("latent_dim", LATENT_DIM)))
    train_base_channels = int(data.get("base_channels", state["train_config"].get("base_channels", BASE_CHANNELS)))

    _stop_event.clear()
    if train_epochs < 1 or train_epochs > 500:
        return jsonify({"error": "epochs must be in [1, 500]"}), 400
    if train_bs < 1 or train_bs > 4096:
        return jsonify({"error": "batch_size must be in [1, 4096]"}), 400
    if train_lr <= 0 or train_lr > 1:
        return jsonify({"error": "learning_rate must be in (0, 1]"}), 400
    if train_patience < 1 or train_patience > 200:
        return jsonify({"error": "patience must be in [1, 200]"}), 400
    if train_quantile <= 0 or train_quantile >= 1:
        return jsonify({"error": "anomaly_quantile must be in (0, 1)"}), 400
    if train_threshold_method not in {"kde_fpr", "mad", "quantile"}:
        return jsonify({"error": "threshold_method must be one of: kde_fpr, mad, quantile"}), 400
    if train_target_fpr <= 0 or train_target_fpr >= 1:
        return jsonify({"error": "threshold_target_fpr must be in (0, 1)"}), 400
    if train_mad_k <= 0 or train_mad_k > 20:
        return jsonify({"error": "threshold_mad_k must be in (0, 20]"}), 400
    if train_latent_l1 <= 0 or train_latent_l1 > 1:
        return jsonify({"error": "latent_l1 must be in (0, 1]"}), 400
    if train_latent_dim < 4 or train_latent_dim > 256:
        return jsonify({"error": "latent_dim must be in [4, 256]"}), 400
    if train_base_channels < 8 or train_base_channels > 256:
        return jsonify({"error": "base_channels must be in [8, 256]"}), 400

    state["train_config"] = {
        "epochs": train_epochs,
        "batch_size": train_bs,
        "learning_rate": train_lr,
        "patience": train_patience,
        "anomaly_quantile": train_quantile,
        "threshold_method": train_threshold_method,
        "threshold_target_fpr": train_target_fpr,
        "threshold_mad_k": train_mad_k,
        "latent_dim": train_latent_dim,
        "latent_l1": train_latent_l1,
        "base_channels": train_base_channels,
    }

    def _run():
        _update_state(status="running", progress=5, log=[], results={})
        try:
            import torch
            from torch.utils.data import DataLoader
            from src.dataset import build_datasets
            from src.autoencoder import ConvAutoencoder, train_autoencoder, compute_threshold
            from src.evaluate import evaluate_anomaly_detection, partial_auc
            from sklearn.metrics import (
                roc_auc_score, roc_curve, confusion_matrix,
                accuracy_score, f1_score, classification_report,
            )
            import matplotlib.pyplot as plt
            import seaborn as sns

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            log(f"Device: {device}  CUDA={torch.cuda.is_available()}")
            if torch.cuda.is_available():
                log(f"GPU: {torch.cuda.get_device_name(0)}")
            log(f"Loading + precomputing spectrograms: {machine_type}…")
            state["progress"] = 8

            def precompute_cb(done, total):
                pct = 8 + int(done / total * 7)
                state["progress"] = pct
                if done <= total:
                    log(f"Preparing dataset files: {done}/{total}")
                else:
                    log(f"Precomputing spectrograms: {done-total}/{total}")

            train_ds, val_ds, test_ds, test_labels, _, norm_stats = build_datasets(
                machine_type, snr_db=snr_db, augment_train=True, progress_cb=precompute_cb, machine_id=machine_id
            )
            log(f"Norm stats (dB min/max): {norm_stats[0]:.2f} / {norm_stats[1]:.2f}")
            log(f"train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}")
            state["progress"] = 15

            # All data is precomputed tensors — num_workers=0 is optimal (no disk I/O)
            bs = train_bs
            train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                                      num_workers=0, pin_memory=(str(device)=="cuda"))
            val_loader   = DataLoader(val_ds,   batch_size=bs, shuffle=False,
                                      num_workers=0, pin_memory=(str(device)=="cuda"))
            test_loader  = DataLoader(test_ds,  batch_size=bs, shuffle=False,
                                      num_workers=0, pin_memory=(str(device)=="cuda"))

            model = ConvAutoencoder(
                latent_dim=train_latent_dim,
                base_channels=train_base_channels,
            ).to(device)
            loaded_from = None
            if train_mode == "finetune":
                registry = _load_registry()
                candidates = [
                    m for m in registry.get("models", [])
                    if m.get("machine_type") == machine_type
                    and (machine_id is None or m.get("machine_id") == machine_id)
                    and int(m.get("snr", snr_db)) == snr_db
                ]
                if base_model_id:
                    candidates = [m for m in candidates if m.get("id") == base_model_id]
                candidates.sort(key=lambda x: x.get("created_at", ""), reverse=True)
                selected = candidates[0] if candidates else None
                if selected:
                    base_model_path = Path(ROOT) / selected["model_path"]
                    if base_model_path.exists():
                        log(f"Finetune from {selected['id']} ({base_model_path.name})")
                        try:
                            _load_model_weights_checked(model, base_model_path, device)
                            loaded_from = selected["id"]
                        except Exception as e:
                            log(f"WARNING finetune base incompatible: {e}. Fallback to new training.")
                    else:
                        log(f"WARNING base model missing: {base_model_path}. Fallback to new training.")
                else:
                    log("WARNING no base model found for finetune. Fallback to new training.")

            total_params = sum(p.numel() for p in model.parameters())
            log(f"ConvAutoencoder  latent_dim={train_latent_dim}  base_channels={train_base_channels} "
                f"latent_l1={train_latent_l1:g}  "
                f"params={total_params:,}  training on {device}…")
            state["progress"] = 18

            # epoch callback for live progress
            save_path = _model_path_for(machine_type, machine_id, snr_db)
            epochs_done = [0]

            def epoch_cb(epoch, t_loss, v_loss):
                epochs_done[0] = epoch
                pct = 18 + int(epoch / train_epochs * 62)
                state["progress"] = min(pct, 80)
                log(f"Epoch {epoch:3d}/{train_epochs} | train={t_loss:.4f} | val={v_loss:.4f}")

            use_amp = (str(device) == "cuda")
            log(f"AMP={'ON' if use_amp else 'OFF'}  device={device}")
            model, history = train_autoencoder(
                model, train_loader, val_loader,
                epochs=train_epochs, lr=train_lr, patience=train_patience,
                device=device, save_path=save_path, cb=epoch_cb, use_amp=use_amp,
                latent_l1=train_latent_l1, stop_event=_stop_event,
            )
            state["progress"] = 80
            if history["train_loss"] and history["val_loss"]:
                log(
                    f"Final losses | train={history['train_loss'][-1]:.6f} "
                    f"val={history['val_loss'][-1]:.6f}"
                )

            log("Computing threshold…")
            threshold_path = _threshold_path_for(machine_type, machine_id, snr_db)
            threshold = compute_threshold(
                model, val_loader, device,
                quantile=train_quantile,
                method=train_threshold_method,
                target_fpr=train_target_fpr,
                mad_k=train_mad_k,
            )
            np.save(threshold_path, threshold)
            norm_stats_path = _norm_stats_path_for(machine_type, machine_id, snr_db)
            _save_norm_stats(norm_stats_path, norm_stats)
            _model_cache[_model_key(machine_type, machine_id, snr_db)] = {
                "model": model, "threshold": threshold, "norm_stats": norm_stats,
            }
            if _model_key(machine_type, machine_id, snr_db) not in state["available_models"]:
                state["available_models"].append(_model_key(machine_type, machine_id, snr_db))
            state["progress"] = 83

            log("Evaluating…")
            eval_res = evaluate_anomaly_detection(model, test_loader, test_labels, device, threshold)
            scores   = eval_res["errors"]
            auc      = float(roc_auc_score(test_labels, scores))
            pauc     = float(partial_auc(test_labels, scores))
            preds    = (scores > threshold).astype(int)
            norm_mean = float(scores[test_labels == 0].mean()) if (test_labels == 0).any() else 0.0
            abn_mean = float(scores[test_labels == 1].mean()) if (test_labels == 1).any() else 0.0
            separation = abn_mean - norm_mean
            log(f"Error separation | normal={norm_mean:.6f} abnormal={abn_mean:.6f} delta={separation:.6f}")
            state["progress"] = 87

            # ── plots ──
            log("Generating plots…")
            norm_scores = scores[test_labels == 0]
            abn_scores  = scores[test_labels == 1]

            # Score distribution
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.hist(norm_scores, bins=60, alpha=0.65, label="Normal",   color="steelblue", density=True)
            ax.hist(abn_scores,  bins=60, alpha=0.65, label="Abnormal", color="tomato",    density=True)
            ax.axvline(threshold, color="black", linestyle="--", lw=2, label=f"thr={threshold:.4f}")
            ax.set_xlabel("Anomaly Score"); ax.set_ylabel("Density")
            ax.set_title(f"Score Distribution — {machine_type}"); ax.legend()
            suffix = _model_key(machine_type, machine_id, snr_db)
            plt.tight_layout(); fig.savefig(RESULTS_DIR / f"scores_{suffix}.png", dpi=150)
            plt.close(fig)

            # ROC
            fpr, tpr, _ = roc_curve(test_labels, scores)
            fig, ax = plt.subplots(figsize=(7, 7))
            ax.plot(fpr, tpr, lw=2, label=f"AUC = {auc:.4f}")
            ax.plot([0,1],[0,1],"k--",lw=1)
            ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
            ax.set_title(f"ROC — {machine_type}"); ax.legend(loc="lower right")
            plt.tight_layout(); fig.savefig(RESULTS_DIR / f"roc_{suffix}.png", dpi=150)
            plt.close(fig)

            # Confusion matrix
            cm = confusion_matrix(test_labels, preds)
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                        xticklabels=["Normal","Anomaly"],
                        yticklabels=["Normal","Anomaly"], ax=ax)
            ax.set_xlabel("Predicted"); ax.set_ylabel("True")
            ax.set_title(f"Confusion Matrix — {machine_type}")
            plt.tight_layout(); fig.savefig(RESULTS_DIR / f"cm_{suffix}.png", dpi=150)
            plt.close(fig)

            # Training history
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(history["train_loss"], label="Train Loss")
            ax.plot(history["val_loss"],   label="Val Loss")
            ax.set_xlabel("Epoch"); ax.set_ylabel("Training Objective")
            ax.set_title(f"Training History — {machine_type}"); ax.legend()
            plt.tight_layout(); fig.savefig(RESULTS_DIR / f"history_{suffix}.png", dpi=150)
            plt.close(fig)

            state["progress"] = 95
            state["results"]["train"] = {
                "machine_type": machine_type,
                "machine_id":   machine_id,
                "snr":          snr_db,
                "train_mode":   train_mode,
                "finetuned_from": loaded_from,
                "auc_roc":      round(auc, 4),
                "pauc":         round(pauc, 4),
                "threshold":    round(float(threshold), 6),
                "train_clips":  len(train_ds),
                "accuracy":     round(float(accuracy_score(test_labels, preds)), 4),
                "f1":           round(float(f1_score(test_labels, preds)), 4),
                "epochs_trained": len(history.get("train_loss", [])),
                "config": {
                    "epochs": train_epochs,
                    "batch_size": train_bs,
                    "learning_rate": train_lr,
                    "patience": train_patience,
                    "anomaly_quantile": train_quantile,
                    "threshold_method": train_threshold_method,
                    "threshold_target_fpr": train_target_fpr,
                    "threshold_mad_k": train_mad_k,
                    "latent_dim": train_latent_dim,
                    "latent_l1": train_latent_l1,
                    "base_channels": train_base_channels,
                },
                "report":       classification_report(test_labels, preds,
                                                      target_names=["Normal","Anomaly"]),
                "normal_error_mean": round(norm_mean, 6),
                "abnormal_error_mean": round(abn_mean, 6),
                "error_delta": round(separation, 6),
            }
            run_record = _register_model(
                machine_type=machine_type,
                machine_id=machine_id,
                snr_db=snr_db,
                threshold=threshold,
                metrics=state["results"]["train"],
                mode=("finetune" if loaded_from else "new"),
                base_model_id=loaded_from,
                epochs_trained=len(history.get("train_loss", [])),
                model_path=save_path,
                threshold_path=threshold_path,
                norm_stats_path=norm_stats_path,
            )
            # snapshot each training run for history/versioning
            hist_model = MODEL_HISTORY_DIR / f"{run_record['id']}.pt"
            hist_thr = MODEL_HISTORY_DIR / f"{run_record['id']}_threshold.npy"
            hist_norm = MODEL_HISTORY_DIR / f"{run_record['id']}_norm.json"
            shutil.copy2(save_path, hist_model)
            shutil.copy2(threshold_path, hist_thr)
            shutil.copy2(norm_stats_path, hist_norm)
            log(f"Saved history model: {run_record['id']}")
            state["progress"] = 100
            log(f"Done! AUC={auc:.4f}  pAUC={pauc:.4f}  thr={threshold:.6f}")

        except Exception as e:
            log(f"ERROR: {e}")
            traceback.print_exc()
        finally:
            _update_state(status="idle")

    threading.Thread(target=_run, daemon=True).start()
    return jsonify({"ok": True})


@app.route("/api/tune", methods=["POST"])
def tune():
    if state["status"] == "running":
        return jsonify({"error": "Busy"}), 409

    data = request.json or {}
    machine_type = data.get("machine_type", "fan")
    machine_id = _normalize_machine_id(data.get("machine_id"))
    snr_db = int(data.get("snr", 6))
    n_trials = int(data.get("trials", 12))
    trial_epochs = int(data.get("trial_epochs", 20))
    trial_patience = int(data.get("trial_patience", 5))
    trial_bs = int(data.get("batch_size", BATCH_SIZE))
    threshold_method = (data.get("threshold_method", THRESHOLD_METHOD) or THRESHOLD_METHOD).strip().lower()
    threshold_target_fpr = float(data.get("threshold_target_fpr", THRESHOLD_TARGET_FPR))
    threshold_mad_k = float(data.get("threshold_mad_k", THRESHOLD_MAD_K))
    threshold_quantile = float(data.get("anomaly_quantile", ANOMALY_QUANTILE))

    _stop_event.clear()
    if n_trials < 3 or n_trials > 200:
        return jsonify({"error": "trials must be in [3, 200]"}), 400
    if trial_epochs < 5 or trial_epochs > 120:
        return jsonify({"error": "trial_epochs must be in [5, 120]"}), 400
    if trial_patience < 2 or trial_patience > 30:
        return jsonify({"error": "trial_patience must be in [2, 30]"}), 400
    if trial_bs < 1 or trial_bs > 4096:
        return jsonify({"error": "batch_size must be in [1, 4096]"}), 400
    if threshold_method not in {"kde_fpr", "mad", "quantile"}:
        return jsonify({"error": "threshold_method must be one of: kde_fpr, mad, quantile"}), 400
    if threshold_target_fpr <= 0 or threshold_target_fpr >= 1:
        return jsonify({"error": "threshold_target_fpr must be in (0, 1)"}), 400
    if threshold_mad_k <= 0 or threshold_mad_k > 20:
        return jsonify({"error": "threshold_mad_k must be in (0, 20]"}), 400
    if threshold_quantile <= 0 or threshold_quantile >= 1:
        return jsonify({"error": "anomaly_quantile must be in (0, 1)"}), 400

    state["tune_config"] = {
        "trials": n_trials,
        "trial_epochs": trial_epochs,
        "trial_patience": trial_patience,
        "threshold_method": threshold_method,
        "threshold_target_fpr": threshold_target_fpr,
        "threshold_mad_k": threshold_mad_k,
        "anomaly_quantile": threshold_quantile,
    }

    def _run():
        _update_state(status="running", progress=4, log=[], results={})
        try:
            import torch
            import optuna
            from torch.utils.data import DataLoader
            from sklearn.metrics import roc_auc_score
            from src.dataset import build_datasets
            from src.autoencoder import ConvAutoencoder, train_autoencoder, compute_threshold

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            log(f"AutoTune start: {machine_type}/{machine_id} @ {snr_db}dB | trials={n_trials}")
            log(f"Device: {device}  CUDA={torch.cuda.is_available()}")

            def precompute_cb(done, total):
                pct = 4 + int(done / max(1, total) * 10)
                state["progress"] = min(15, pct)

            train_ds, val_ds, test_ds, test_labels, _, norm_stats = build_datasets(
                machine_type, snr_db=snr_db, augment_train=True, progress_cb=precompute_cb, machine_id=machine_id
            )
            tune_ds, tune_labels = _build_tune_split(test_ds, test_labels, rng_seed=42, keep_ratio=0.4)
            state["progress"] = 16

            base_train_cfg = deepcopy(state["train_config"])
            log(
                f"Tune dataset | train={len(train_ds)} val={len(val_ds)} "
                f"tune={len(tune_ds)} (norm={int((tune_labels==0).sum())}, abn={int((tune_labels==1).sum())})"
            )

            val_loader_cache = DataLoader(
                val_ds, batch_size=trial_bs, shuffle=False, num_workers=0, pin_memory=(str(device) == "cuda")
            )
            tune_loader_cache = DataLoader(
                tune_ds, batch_size=trial_bs, shuffle=False, num_workers=0, pin_memory=(str(device) == "cuda")
            )

            def objective(trial):
                latent_dim = trial.suggest_categorical("latent_dim", [8, 16, 24])
                base_channels = trial.suggest_categorical("base_channels", [32, 64])
                lr = trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True)
                latent_l1 = trial.suggest_float("latent_l1", 5e-4, 1e-2, log=True)
                mad_k = trial.suggest_float("threshold_mad_k", 1.5, 3.5)
                q = trial.suggest_float("anomaly_quantile", 0.85, 0.99)
                method = trial.suggest_categorical("threshold_method", ["mad", "quantile"])

                train_loader = DataLoader(
                    train_ds, batch_size=trial_bs, shuffle=True, num_workers=0, pin_memory=(str(device) == "cuda")
                )

                model = ConvAutoencoder(latent_dim=latent_dim, base_channels=base_channels).to(device)
                use_amp = (str(device) == "cuda")
                try:
                    def epoch_cb(epoch, t_loss, v_loss):
                        trial.report(-v_loss, step=epoch)
                        if trial.should_prune():
                            raise optuna.TrialPruned()

                    model, _ = train_autoencoder(
                        model, train_loader, val_loader_cache,
                        epochs=trial_epochs, lr=lr, patience=trial_patience,
                        device=device, save_path=None, cb=epoch_cb, use_amp=use_amp,
                        latent_l1=latent_l1,
                    )

                    threshold = compute_threshold(
                        model, val_loader_cache, device,
                        quantile=q if method == "quantile" else threshold_quantile,
                        method=method,
                        target_fpr=threshold_target_fpr,
                        mad_k=mad_k if method == "mad" else threshold_mad_k,
                    )

                    model.eval()
                    scores = []
                    with torch.no_grad():
                        for batch in tune_loader_cache:
                            x = batch[0] if isinstance(batch, (list, tuple)) else batch
                            x = x.to(device, non_blocking=True)
                            errs = model.reconstruction_error(x).detach().cpu().numpy()
                            scores.extend(errs.tolist())
                    scores = np.asarray(scores, dtype=np.float64)
                    auc = float(roc_auc_score(tune_labels, scores))
                    norm_mean = float(scores[tune_labels == 0].mean())
                    abn_mean = float(scores[tune_labels == 1].mean())
                    delta = abn_mean - norm_mean
                    pred_rate = float((scores > threshold).mean())

                    score = auc + 0.2 * max(0.0, delta) + 0.05 * min(pred_rate, 0.2)
                    trial.set_user_attr("auc", auc)
                    trial.set_user_attr("delta", delta)
                    trial.set_user_attr("threshold", float(threshold))
                    trial.set_user_attr("pred_rate", pred_rate)
                    return score
                finally:
                    del model
                    if str(device) == "cuda":
                        torch.cuda.empty_cache()

            pruner = optuna.pruners.MedianPruner(n_startup_trials=max(3, n_trials // 5), n_warmup_steps=5, interval_steps=2)
            sampler = optuna.samplers.TPESampler(seed=42)
            study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)

            done = {"n": 0}

            def trial_cb(study_obj, trial_obj):
                done["n"] += 1
                pct = 16 + int(done["n"] / max(1, n_trials) * 68)
                state["progress"] = min(84, pct)
                val = trial_obj.value if trial_obj.value is not None else float("nan")
                auc_attr = trial_obj.user_attrs.get("auc", float("nan"))
                delta_attr = trial_obj.user_attrs.get("delta", float("nan"))
                log(
                    f"Tune trial {done['n']}/{n_trials} [{trial_obj.state.name}] "
                    f"score={val:.5f} auc={auc_attr:.5f} delta={delta_attr:.6f}"
                )

            study.optimize(objective, n_trials=n_trials, callbacks=[trial_cb])

            best = study.best_trial
            best_params = dict(best.params)
            best_method = best_params.get("threshold_method", threshold_method)
            best_quantile = float(best_params.get("anomaly_quantile", threshold_quantile))
            best_mad_k = float(best_params.get("threshold_mad_k", threshold_mad_k))

            # Apply best params to train defaults.
            state["train_config"].update({
                "learning_rate": float(best_params.get("learning_rate", base_train_cfg["learning_rate"])),
                "batch_size": trial_bs,
                "anomaly_quantile": best_quantile,
                "threshold_method": best_method,
                "threshold_target_fpr": threshold_target_fpr,
                "threshold_mad_k": best_mad_k,
                "latent_dim": int(best_params.get("latent_dim", base_train_cfg.get("latent_dim", LATENT_DIM))),
                "latent_l1": float(best_params.get("latent_l1", base_train_cfg.get("latent_l1", LATENT_L1))),
                "base_channels": int(best_params.get("base_channels", base_train_cfg.get("base_channels", BASE_CHANNELS))),
            })

            state["results"]["tune"] = {
                "machine_type": machine_type,
                "machine_id": machine_id,
                "snr": snr_db,
                "trials": n_trials,
                "best_score": round(float(best.value), 6),
                "best_auc": round(float(best.user_attrs.get("auc", 0.0)), 6),
                "best_error_delta": round(float(best.user_attrs.get("delta", 0.0)), 6),
                "best_threshold": round(float(best.user_attrs.get("threshold", 0.0)), 6),
                "best_pred_rate": round(float(best.user_attrs.get("pred_rate", 0.0)), 6),
                "best_params": best_params,
                "applied_train_config": deepcopy(state["train_config"]),
                "norm_stats": [round(float(norm_stats[0]), 4), round(float(norm_stats[1]), 4)],
            }

            log(f"AutoTune done. best_auc={best.user_attrs.get('auc', 0.0):.5f} delta={best.user_attrs.get('delta', 0.0):.6f}")
            state["progress"] = 100
        except Exception as e:
            log(f"ERROR: {e}")
            traceback.print_exc()
        finally:
            _update_state(status="idle")

    threading.Thread(target=_run, daemon=True).start()
    return jsonify({"ok": True})


@app.route("/api/autopilot", methods=["POST"])
def autopilot():
    if state["status"] == "running":
        return jsonify({"error": "Busy"}), 409

    data = request.json or {}
    machine_type = data.get("machine_type", "fan")
    machine_id = _normalize_machine_id(data.get("machine_id"))
    snr_db = int(data.get("snr", 6))
    target_auc = float(data.get("target_auc", 0.89))
    max_rounds = int(data.get("max_rounds", 3))
    trials_per_round = int(data.get("trials_per_round", 10))
    trial_epochs = int(data.get("trial_epochs", 18))
    full_epochs = int(data.get("full_epochs", min(70, state["train_config"].get("epochs", EPOCHS))))

    _stop_event.clear()
    if target_auc <= 0.5 or target_auc > 0.99:
        return jsonify({"error": "target_auc must be in (0.5, 0.99]"}), 400
    if max_rounds < 1 or max_rounds > 10:
        return jsonify({"error": "max_rounds must be in [1, 10]"}), 400
    if trials_per_round < 3 or trials_per_round > 80:
        return jsonify({"error": "trials_per_round must be in [3, 80]"}), 400
    if trial_epochs < 5 or trial_epochs > 120:
        return jsonify({"error": "trial_epochs must be in [5, 120]"}), 400
    if full_epochs < 10 or full_epochs > 500:
        return jsonify({"error": "full_epochs must be in [10, 500]"}), 400

    def _run():
        _update_state(status="running", progress=2, log=[], results={})
        try:
            import torch
            import optuna
            from torch.utils.data import DataLoader
            from sklearn.metrics import roc_auc_score
            from src.dataset import build_datasets
            from src.autoencoder import ConvAutoencoder, train_autoencoder, compute_threshold

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            log(f"AutoPilot start | target_auc={target_auc:.2f} | rounds={max_rounds}")
            log(f"Task: {machine_type}/{machine_id} @ {snr_db}dB | device={device}")

            # Build datasets once for speed.
            def precompute_cb(done, total):
                state["progress"] = min(12, 2 + int(done / max(1, total) * 10))

            train_ds, val_ds, test_ds, test_labels, _, norm_stats = build_datasets(
                machine_type, snr_db=snr_db, augment_train=True, progress_cb=precompute_cb, machine_id=machine_id
            )
            tune_ds, tune_labels = _build_tune_split(test_ds, test_labels, rng_seed=42, keep_ratio=0.4)

            batch_size = int(state["train_config"].get("batch_size", BATCH_SIZE))
            val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=(str(device) == "cuda"))
            tune_loader = DataLoader(tune_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=(str(device) == "cuda"))
            test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=(str(device) == "cuda"))

            round_results = []
            best_round = None

            for round_idx in range(1, max_rounds + 1):
                log(f"[AutoPilot] round {round_idx}/{max_rounds}: tuning...")
                state["progress"] = min(85, 12 + int((round_idx - 1) / max_rounds * 70))

                # Adapt search space by previous rounds.
                if round_idx == 1:
                    latent_candidates = [8, 16, 24]
                    channel_candidates = [32, 64]
                    l1_low, l1_high = 5e-4, 1e-2
                else:
                    prev = best_round["best_params"]
                    base_latent = int(prev.get("latent_dim", 16))
                    base_ch = int(prev.get("base_channels", 32))
                    latent_candidates = sorted(set([max(8, base_latent // 2), base_latent, min(48, base_latent * 2)]))
                    channel_candidates = sorted(set([max(16, base_ch // 2), base_ch, min(128, base_ch * 2)]))
                    l1_low, l1_high = 1e-4, 2e-2

                def objective(trial):
                    latent_dim = trial.suggest_categorical("latent_dim", latent_candidates)
                    base_channels = trial.suggest_categorical("base_channels", channel_candidates)
                    lr = trial.suggest_float("learning_rate", 1e-5, 7e-4, log=True)
                    latent_l1 = trial.suggest_float("latent_l1", l1_low, l1_high, log=True)
                    method = trial.suggest_categorical("threshold_method", ["mad", "quantile"])
                    mad_k = trial.suggest_float("threshold_mad_k", 1.5, 4.0)
                    q = trial.suggest_float("anomaly_quantile", 0.80, 0.99)

                    train_loader = DataLoader(
                        train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=(str(device) == "cuda")
                    )
                    model = ConvAutoencoder(latent_dim=latent_dim, base_channels=base_channels).to(device)
                    try:
                        def epoch_cb(epoch, t_loss, v_loss):
                            trial.report(-v_loss, step=epoch)
                            if trial.should_prune():
                                raise optuna.TrialPruned()

                        model, _ = train_autoencoder(
                            model, train_loader, val_loader,
                            epochs=trial_epochs, lr=lr, patience=max(5, trial_epochs // 4),
                            device=device, save_path=None, cb=epoch_cb,
                            use_amp=(str(device) == "cuda"), latent_l1=latent_l1,
                        )

                        threshold = compute_threshold(
                            model, val_loader, device,
                            quantile=q if method == "quantile" else state["train_config"].get("anomaly_quantile", ANOMALY_QUANTILE),
                            method=method,
                            target_fpr=state["train_config"].get("threshold_target_fpr", THRESHOLD_TARGET_FPR),
                            mad_k=mad_k if method == "mad" else state["train_config"].get("threshold_mad_k", THRESHOLD_MAD_K),
                        )

                        model.eval()
                        scores = []
                        with torch.no_grad():
                            for batch in tune_loader:
                                x = batch[0] if isinstance(batch, (list, tuple)) else batch
                                x = x.to(device, non_blocking=True)
                                scores.extend(model.reconstruction_error(x).detach().cpu().numpy().tolist())
                        scores = np.asarray(scores, dtype=np.float64)
                        auc = float(roc_auc_score(tune_labels, scores))
                        delta = float(scores[tune_labels == 1].mean() - scores[tune_labels == 0].mean())
                        pred_rate = float((scores > threshold).mean())
                        score = auc + 0.20 * max(delta, 0.0) + 0.05 * min(pred_rate, 0.2)
                        trial.set_user_attr("auc", auc)
                        trial.set_user_attr("delta", delta)
                        trial.set_user_attr("threshold", float(threshold))
                        return score
                    finally:
                        del model
                        if str(device) == "cuda":
                            torch.cuda.empty_cache()

                study = optuna.create_study(
                    direction="maximize",
                    sampler=optuna.samplers.TPESampler(seed=42 + round_idx),
                    pruner=optuna.pruners.MedianPruner(
                        n_startup_trials=max(3, trials_per_round // 5), n_warmup_steps=5, interval_steps=2
                    ),
                )
                study.optimize(objective, n_trials=trials_per_round)
                best = study.best_trial
                best_params = dict(best.params)
                log(
                    f"[AutoPilot] round {round_idx} tune best: "
                    f"auc={best.user_attrs.get('auc', 0.0):.4f} delta={best.user_attrs.get('delta', 0.0):.6f}"
                )

                # Full training with best params
                log(f"[AutoPilot] round {round_idx}: full training...")
                model = ConvAutoencoder(
                    latent_dim=int(best_params.get("latent_dim", state["train_config"].get("latent_dim", LATENT_DIM))),
                    base_channels=int(best_params.get("base_channels", state["train_config"].get("base_channels", BASE_CHANNELS))),
                ).to(device)
                save_path = _model_path_for(machine_type, machine_id, snr_db)
                model, history = train_autoencoder(
                    model,
                    DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=(str(device) == "cuda")),
                    val_loader,
                    epochs=full_epochs,
                    lr=float(best_params.get("learning_rate", state["train_config"].get("learning_rate", LEARNING_RATE))),
                    patience=int(state["train_config"].get("patience", PATIENCE)),
                    device=device,
                    save_path=save_path,
                    cb=lambda e, tl, vl: None,
                    use_amp=(str(device) == "cuda"),
                    latent_l1=float(best_params.get("latent_l1", state["train_config"].get("latent_l1", LATENT_L1))),
                )
                threshold = compute_threshold(
                    model, val_loader, device,
                    quantile=float(best_params.get("anomaly_quantile", state["train_config"].get("anomaly_quantile", ANOMALY_QUANTILE))),
                    method=(best_params.get("threshold_method", state["train_config"].get("threshold_method", THRESHOLD_METHOD))),
                    target_fpr=float(state["train_config"].get("threshold_target_fpr", THRESHOLD_TARGET_FPR)),
                    mad_k=float(best_params.get("threshold_mad_k", state["train_config"].get("threshold_mad_k", THRESHOLD_MAD_K))),
                )
                threshold_path = _threshold_path_for(machine_type, machine_id, snr_db)
                np.save(threshold_path, threshold)
                norm_stats_path = _norm_stats_path_for(machine_type, machine_id, snr_db)
                _save_norm_stats(norm_stats_path, norm_stats)

                # Test metrics
                scores = []
                model.eval()
                with torch.no_grad():
                    for batch in test_loader:
                        x = batch[0] if isinstance(batch, (list, tuple)) else batch
                        x = x.to(device, non_blocking=True)
                        scores.extend(model.reconstruction_error(x).detach().cpu().numpy().tolist())
                scores = np.asarray(scores, dtype=np.float64)
                auc = float(roc_auc_score(test_labels, scores))
                pauc = float(_partial_auc(test_labels, scores))
                norm_mean = float(scores[test_labels == 0].mean())
                abn_mean = float(scores[test_labels == 1].mean())
                delta = abn_mean - norm_mean

                rr = {
                    "round": round_idx,
                    "auc_roc": round(auc, 6),
                    "pauc": round(pauc, 6),
                    "error_delta": round(delta, 6),
                    "threshold": round(float(threshold), 6),
                    "best_params": best_params,
                    "epochs_trained": len(history.get("train_loss", [])),
                }
                round_results.append(rr)
                if best_round is None or auc > best_round["auc_roc"]:
                    best_round = {"auc_roc": auc, "best_params": best_params, "round": round_idx, "threshold": threshold}

                # Update train defaults with best of this round.
                state["train_config"].update({
                    "learning_rate": float(best_params.get("learning_rate", state["train_config"]["learning_rate"])),
                    "anomaly_quantile": float(best_params.get("anomaly_quantile", state["train_config"]["anomaly_quantile"])),
                    "threshold_method": best_params.get("threshold_method", state["train_config"]["threshold_method"]),
                    "threshold_mad_k": float(best_params.get("threshold_mad_k", state["train_config"]["threshold_mad_k"])),
                    "latent_dim": int(best_params.get("latent_dim", state["train_config"]["latent_dim"])),
                    "latent_l1": float(best_params.get("latent_l1", state["train_config"]["latent_l1"])),
                    "base_channels": int(best_params.get("base_channels", state["train_config"]["base_channels"])),
                })

                _model_cache[_model_key(machine_type, machine_id, snr_db)] = {
                    "model": model, "threshold": threshold, "norm_stats": norm_stats,
                }
                key = _model_key(machine_type, machine_id, snr_db)
                if key not in state["available_models"]:
                    state["available_models"].append(key)

                if auc >= target_auc:
                    log(f"[AutoPilot] target reached at round {round_idx}: AUC={auc:.4f}")
                    break
                log(f"[AutoPilot] round {round_idx} done: AUC={auc:.4f} (< {target_auc:.2f})")

            state["results"]["autopilot"] = {
                "machine_type": machine_type,
                "machine_id": machine_id,
                "snr": snr_db,
                "target_auc": target_auc,
                "rounds_ran": len(round_results),
                "round_results": round_results,
                "best_round": best_round["round"] if best_round else None,
                "best_auc": round(float(best_round["auc_roc"]), 6) if best_round else 0.0,
                "best_params": best_round["best_params"] if best_round else {},
                "best_threshold": round(float(best_round["threshold"]), 6) if best_round else 0.0,
                "status": "success" if best_round and best_round["auc_roc"] >= target_auc else "target_not_reached",
            }
            state["progress"] = 100
            log(
                f"[AutoPilot] done | best_auc={state['results']['autopilot']['best_auc']:.4f} "
                f"| status={state['results']['autopilot']['status']}"
            )
        except Exception as e:
            log(f"ERROR: {e}")
            traceback.print_exc()
        finally:
            _update_state(status="idle")

    threading.Thread(target=_run, daemon=True).start()
    return jsonify({"ok": True})


@app.route("/api/train_baseline", methods=["POST"])
def train_baseline():
    """Train the baseline DenseAutoencoder (MIMII style, MSE reconstruction)."""
    if state["status"] == "running":
        return jsonify({"error": "Busy"}), 409
    data         = request.json or {}
    machine_type = data.get("machine_type", "fan")
    machine_id   = _normalize_machine_id(data.get("machine_id"))
    snr_db       = int(data.get("snr", 6))
    train_epochs = int(data.get("epochs", EPOCHS))
    train_bs     = int(data.get("batch_size", BATCH_SIZE))
    train_lr     = float(data.get("learning_rate", LEARNING_RATE))
    train_patience = int(data.get("patience", PATIENCE))

    _stop_event.clear()
    if train_epochs < 1 or train_epochs > 500:
        return jsonify({"error": "epochs must be in [1, 500]"}), 400
    if train_bs < 1 or train_bs > 8192:
        return jsonify({"error": "batch_size must be in [1, 8192]"}), 400
    if train_lr <= 0 or train_lr > 1:
        return jsonify({"error": "learning_rate must be in (0, 1]"}), 400
    if train_patience < 1 or train_patience > 200:
        return jsonify({"error": "patience must be in [1, 200]"}), 400

    def _run():
        _update_state(status="running", progress=5, log=[], results={})
        try:
            import torch
            from torch.utils.data import DataLoader
            from src.dataset import build_baseline_datasets
            from src.autoencoder import DenseAutoencoder, train_dense_autoencoder
            from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, accuracy_score, f1_score, classification_report

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            log(f"[Baseline AE] Device: {device}")
            log(f"Loading data: {machine_type} @ {snr_db}dB...")
            state["progress"] = 10

            def precompute_cb(done, total):
                pct = 10 + int(done / max(1, total) * 5)
                state["progress"] = min(20, pct)

            train_ds, val_ds, test_ds, test_labels = build_baseline_datasets(
                machine_type, snr_db=snr_db, progress_cb=precompute_cb, machine_id=machine_id,
                n_mels=BASELINE_N_MELS, frames=BASELINE_FRAMES,
                n_fft=BASELINE_N_FFT, hop_length=BASELINE_HOP_LENGTH, power=BASELINE_POWER,
            )
            log(f"train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}")
            state["progress"] = 20

            input_dim = BASELINE_N_MELS * BASELINE_FRAMES
            train_loader = DataLoader(train_ds, batch_size=train_bs, shuffle=True, num_workers=0)
            val_loader   = DataLoader(val_ds,   batch_size=train_bs, shuffle=False, num_workers=0)
            test_loader  = DataLoader(test_ds,  batch_size=train_bs, shuffle=False, num_workers=0)

            model = DenseAutoencoder(input_dim=input_dim).to(device)
            n_params = sum(p.numel() for p in model.parameters())
            log(f"DenseAutoencoder ({n_params:,} params) input_dim={input_dim} — training...")
            state["progress"] = 25

            save_path = MODEL_DIR / f"dense_ae_{_model_key(machine_type, machine_id, snr_db)}.pt"

            def epoch_cb(epoch, t_loss, v_loss):
                pct = 25 + int(epoch / train_epochs * 50)
                state["progress"] = min(pct, 75)
                log(f"Epoch {epoch:3d}/{train_epochs} | train={t_loss:.6f} | val={v_loss:.6f}")

            model, history = train_dense_autoencoder(
                model, train_loader, val_loader,
                epochs=train_epochs, lr=train_lr, patience=train_patience,
                device=device, save_path=save_path, cb=epoch_cb,
            )
            state["progress"] = 80

            # Evaluate
            log("Evaluating...")
            model.eval()
            scores = []
            with torch.no_grad():
                for batch in test_loader:
                    x = batch[0] if isinstance(batch, (list, tuple)) else batch
                    x = x.to(device)
                    err = model.reconstruction_error(x)
                    scores.extend(err.cpu().numpy().tolist())
            scores = np.array(scores, dtype=np.float64)
            auc = float(roc_auc_score(test_labels, scores))
            pauc = float(_partial_auc(test_labels, scores))

            # Threshold from validation error distribution (quantile)
            val_scores = []
            with torch.no_grad():
                for batch in val_loader:
                    x = batch[0] if isinstance(batch, (list, tuple)) else batch
                    x = x.to(device)
                    val_scores.extend(model.reconstruction_error(x).cpu().numpy().tolist())
            threshold = float(np.quantile(val_scores, 0.90))
            preds = (scores > threshold).astype(int)

            norm_mean = float(scores[test_labels == 0].mean()) if (test_labels == 0).any() else 0.0
            abn_mean = float(scores[test_labels == 1].mean()) if (test_labels == 1).any() else 0.0
            log(f"Baseline AE | AUC={auc:.4f} pAUC={pauc:.4f} thr={threshold:.6f} "
                f"norm_err={norm_mean:.6f} abn_err={abn_mean:.6f}")

            state["progress"] = 90
            log("Generating plots...")

            import matplotlib.pyplot as plt
            import seaborn as sns
            suffix = f"dense_{_model_key(machine_type, machine_id, snr_db)}"

            # Score distribution
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.hist(scores[test_labels == 0], bins=60, alpha=0.65, label="Normal", color="steelblue", density=True)
            ax.hist(scores[test_labels == 1], bins=60, alpha=0.65, label="Abnormal", color="tomato", density=True)
            ax.axvline(threshold, color="black", linestyle="--", lw=2, label=f"thr={threshold:.4f}")
            ax.set_xlabel("MSE Reconstruction Error"); ax.set_ylabel("Density")
            ax.set_title(f"Baseline AE — {machine_type}"); ax.legend()
            plt.tight_layout(); fig.savefig(RESULTS_DIR / f"scores_{suffix}.png", dpi=150)
            plt.close(fig)

            # ROC
            fpr, tpr, _ = roc_curve(test_labels, scores)
            fig, ax = plt.subplots(figsize=(7, 7))
            ax.plot(fpr, tpr, lw=2, label=f"AUC = {auc:.4f}")
            ax.plot([0, 1], [0, 1], "k--", lw=1)
            ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
            ax.set_title(f"Baseline AE ROC — {machine_type}"); ax.legend(loc="lower right")
            plt.tight_layout(); fig.savefig(RESULTS_DIR / f"roc_{suffix}.png", dpi=150)
            plt.close(fig)

            # Training history
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(history["train_loss"], label="Train Loss")
            ax.plot(history["val_loss"], label="Val Loss")
            ax.set_xlabel("Epoch"); ax.set_ylabel("MSE")
            ax.set_title(f"Baseline AE Training — {machine_type}"); ax.legend()
            plt.tight_layout(); fig.savefig(RESULTS_DIR / f"history_{suffix}.png", dpi=150)
            plt.close(fig)

            state["results"]["baseline"] = {
                "machine_type": machine_type,
                "machine_id": machine_id,
                "snr": snr_db,
                "model": "DenseAutoencoder (baseline)",
                "auc_roc": round(auc, 4),
                "pauc": round(pauc, 4),
                "threshold": round(threshold, 6),
                "accuracy": round(float(accuracy_score(test_labels, preds)), 4),
                "f1": round(float(f1_score(test_labels, preds)), 4),
                "epochs_trained": len(history.get("train_loss", [])),
                "normal_error_mean": round(norm_mean, 6),
                "abnormal_error_mean": round(abn_mean, 6),
                "error_delta": round(abn_mean - norm_mean, 6),
            }
            state["progress"] = 100
            log(f"Baseline AE done! AUC={auc:.4f}  thr={threshold:.6f}")

        except Exception as e:
            log(f"ERROR: {e}")
            traceback.print_exc()
        finally:
            _update_state(status="idle")

    threading.Thread(target=_run, daemon=True).start()
    return jsonify({"ok": True})


@app.route("/api/check_upload", methods=["POST"])
def check_upload():
    if state["status"] == "running":
        return jsonify({"error": "Busy"}), 409

    files = request.files.getlist("files")
    if not files or not files[0].filename:
        return jsonify({"error": "No files uploaded"}), 400

    machine_type  = request.form.get("machine_type", "fan")
    machine_id    = _normalize_machine_id(request.form.get("machine_id"))
    model_id_raw  = (request.form.get("model_id", "") or "").strip()
    threshold_raw = request.form.get("threshold", "").strip()

    snr_raw = (request.form.get("snr") or "").strip()
    snr_db = int(snr_raw) if snr_raw else None
    cache_base_key = _model_key(machine_type, machine_id, snr_db) if snr_db is not None else _model_key(machine_type, machine_id)
    cache_key = f"{cache_base_key}:{model_id_raw or 'latest'}"
    cached = _model_cache.get(cache_key) or _model_cache.get(cache_base_key)
    if not cached:
        selected_path = None
        selected_thr_path = None
        selected_norm_path = None
        if model_id_raw:
            reg = _load_registry()
            rec = next((
                m for m in reg.get("models", [])
                if m.get("id") == model_id_raw
                and m.get("machine_type") == machine_type
                and (machine_id is None or m.get("machine_id") == machine_id)
            ), None)
            if not rec:
                return jsonify({"error": f"Model id '{model_id_raw}' not found for '{machine_type}'/{machine_id or 'all'}."}), 400
            selected_path = Path(ROOT) / rec["model_path"]
            selected_thr_path = Path(ROOT) / rec["threshold_path"]
            selected_norm_path = Path(ROOT) / rec["norm_stats_path"] if rec.get("norm_stats_path") else None
            if snr_db is None:
                snr_db = int(rec.get("snr", 6))
        else:
            if snr_db is None:
                latest = _latest_model_record(machine_type, machine_id=machine_id)
                if latest is None:
                    return jsonify({"error": f"No model for '{machine_type}'/{machine_id or 'all'}. Train first."}), 400
                snr_db = int(latest.get("snr", 6))
            selected_path = _model_path_for(machine_type, machine_id, snr_db)
            selected_thr_path = _threshold_path_for(machine_type, machine_id, snr_db)
            selected_norm_path = _norm_stats_path_for(machine_type, machine_id, snr_db)

        if not selected_path.exists():
            return jsonify({"error": f"No model for '{machine_type}'/{machine_id or 'all'}. Train first."}), 400

        import torch
        from src.autoencoder import ConvAutoencoder
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model  = ConvAutoencoder().to(device)
        try:
            _load_model_weights_checked(model, selected_path, device)
        except Exception as e:
            return jsonify({"error": f"Incompatible model checkpoint with current config: {e}"}), 400
        model.eval()
        threshold = float(np.load(selected_thr_path))
        norm_stats = _load_norm_stats(selected_norm_path) if selected_norm_path else None
        if norm_stats is None:
            log(f"WARNING: no norm stats for this model, scores may be inaccurate.")
        cached = {"model": model, "threshold": threshold, "norm_stats": norm_stats}
        _model_cache[cache_key] = cached
        _model_cache[cache_base_key] = cached

    model      = cached["model"]
    default_t  = cached["threshold"]
    norm_stats = cached.get("norm_stats")
    try:
        threshold = float(threshold_raw) if threshold_raw else default_t
    except ValueError:
        return jsonify({"error": "Threshold must be a number"}), 400

    saved = []
    for f in files:
        safe = secure_filename(f.filename or "audio.wav")
        tmp  = tempfile.NamedTemporaryFile(delete=False, suffix=Path(safe).suffix or ".wav")
        f.save(tmp.name)
        saved.append((safe, tmp.name))

    def _run():
        _update_state(status="running", progress=10, log=[], results={})
        try:
            import torch
            from src.dataset import load_audio, window_signal, extract_mel_spectrogram
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            log(f"Checking {len(saved)} file(s) | {machine_type}/{machine_id or 'all'} | thr={threshold:.6f}")

            items, n_anom = [], 0
            for name, path in saved:
                y       = load_audio(Path(path))
                windows = window_signal(y)
                if not windows:
                    items.append({"name": name, "score": 0.0, "is_anomaly": False})
                    continue
                tensors = torch.stack([
                    torch.from_numpy(extract_mel_spectrogram(w, norm_stats=norm_stats)).float().unsqueeze(0)
                    for w in windows
                ]).to(device)
                with torch.no_grad():
                    errs = model.reconstruction_error(tensors).cpu().numpy()
                score   = float(errs.mean())
                is_anom = score > threshold
                n_anom += int(is_anom)
                items.append({"name": name, "score": round(score, 6), "is_anomaly": bool(is_anom)})

            state["results"]["upload_check"] = {
                "machine_type":  machine_type,
                "threshold":     round(threshold, 6),
                "total_files":   len(saved),
                "anomaly_count": n_anom,
                "items":         items,
            }
            state["progress"] = 100
            log(f"Done. Anomalies: {n_anom}/{len(saved)}")
        except Exception as e:
            log(f"ERROR: {e}")
            traceback.print_exc()
        finally:
            _update_state(status="idle")
            for _, p in saved:
                try: Path(p).unlink()
                except OSError: pass

    threading.Thread(target=_run, daemon=True).start()
    return jsonify({"ok": True})


# ── Deploy endpoint (for agent auto-update) ──────────────────────────────
DEPLOY_TOKEN = os.environ.get("DEPLOY_TOKEN", "")
DEPLOY_ALLOWED = bool(DEPLOY_TOKEN)


@app.route("/api/deploy", methods=["POST"])
def deploy():
    if not DEPLOY_ALLOWED:
        return jsonify({"error": "Deploy disabled. Set DEPLOY_TOKEN env var."}), 403
    token = request.headers.get("X-Deploy-Token", "")
    if token != DEPLOY_TOKEN:
        return jsonify({"error": "Unauthorized"}), 401
    if state["status"] == "running":
        return jsonify({"error": "Busy"}), 409

    data = request.json or {}
    do_pull = data.get("pull", True)
    do_restart = data.get("restart", True)

    def _run():
        _update_state(status="running", progress=0, log=[], results={})
        try:
            import subprocess, sys
            log("Deploy: pulling latest code...")
            if do_pull:
                r = subprocess.run(["git", "pull", "origin", "main"],
                                   capture_output=True, text=True, timeout=60)
                log(r.stdout.strip())
                if r.returncode != 0:
                    log(f"ERROR: git pull failed: {r.stderr.strip()}")
                    _update_state(status="idle")
                    return
                log("Deploy: pull OK")
            _update_state(progress=50)
            log("Deploy: installing deps...")
            subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt", "-q"],
                           capture_output=True, timeout=120)
            _update_state(progress=80)
            if do_restart:
                log("Deploy: restarting server...")
                os._exit(0)
        except subprocess.TimeoutExpired:
            log("ERROR: Deploy timed out")
            _update_state(status="idle")
        except Exception as e:
            log(f"ERROR: {e}")
            traceback.print_exc()
            _update_state(status="idle")

    threading.Thread(target=_run, daemon=True).start()
    return jsonify({"ok": True, "message": "Deploy started"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--public", action="store_true")
    args = parser.parse_args()

    _preload()

    if args.public:
        try:
            import ngrok  # pip install ngrok
            listener  = ngrok.forward(PORT, authtoken_from_env=True)
            public_url = listener.url()
            logger.info("=" * 52)
            logger.info(f"  PUBLIC URL: {public_url}")
            logger.info("=" * 52)
            print(f"\n  PUBLIC URL: {public_url}\n")
        except ImportError:
            logger.warning("ngrok not installed. Run: pip install ngrok")
        except Exception as e:
            logger.warning(f"ngrok failed: {e}")

    try:
        app.run(host="0.0.0.0", port=PORT, debug=False)
    except KeyboardInterrupt:
        pass
