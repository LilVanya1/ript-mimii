#!/usr/bin/env python
"""
 @file   gui.py
 @brief  Web GUI for MIMII baseline — train, view stats, test.
"""
import os
import sys
import glob
import json
import threading
import queue
import time
import logging as py_logging

import numpy as np
import librosa
import yaml
from flask import Flask, render_template, request, jsonify
from sklearn import metrics
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split

from baseline import (
    dataset_generator, list_to_vector_array, file_to_vector_array,
    list_to_patch_array, file_to_patches, audio_to_patches,
    load_pickle, save_pickle, Autoencoder, VAE, CNNAutoencoder,
)

# Suppress Flask's default logging
py_logging.getLogger("werkzeug").setLevel(py_logging.WARNING)

app = Flask(__name__)

state = {
    "training": False,
    "current": None,
    "progress": [],
    "log_queue": queue.Queue(maxsize=2000),
    "results": {},
}

param = None


def load_config():
    global param
    with open("baseline.yaml") as f:
        param = yaml.safe_load(f)


def find_target_dirs(base):
    """Find all directories containing normal/ and abnormal/ subdirs (any depth)."""
    base = os.path.abspath(base)
    targets = []
    for root, dirs, files in os.walk(base):
        if 'normal' in dirs and 'abnormal' in dirs:
            targets.append(root)
            dirs[:] = [d for d in dirs if d not in ('normal', 'abnormal')]
    return sorted(targets)


def get_available_datasets():
    load_config()
    base = os.path.abspath(param["base_directory"])
    if not os.path.exists(base):
        return []
    dirs = find_target_dirs(base)
    datasets = []
    for d in dirs:
        rel = os.path.relpath(d, base).replace(os.sep, '/')
        parts = rel.split('/')
        if len(parts) >= 3:
            # MIMII standard: {db}/{machine_type}/{machine_id}
            db = parts[-3]
            machine_type = parts[-2]
            machine_id = parts[-1]
        elif len(parts) == 2:
            # combined: {type}_{db}/{machine_id}
            machine_id = parts[-1]
            combined = parts[-2]
            machine_type = combined.split('_')[-1]
            db = '_'.join(combined.split('_')[:-1]) if '_' in combined else combined
        else:
            machine_id = parts[-1]
            machine_type = machine_id
            db = "unknown"
        datasets.append({
            "path": d,
            "db": db,
            "machine_type": machine_type,
            "machine_id": machine_id,
            "key": f"{machine_type}_{machine_id}_{db}",
        })
    return datasets


def get_trained_models():
    load_config()
    model_dir = param["model_directory"]
    if not os.path.exists(model_dir):
        return []
    files = sorted(glob.glob(f"{model_dir}/model_*.pth"))
    models = []
    for f in files:
        name = os.path.basename(f).replace("model_", "").replace(".pth", "")
        mtime = os.path.getmtime(f)
        models.append({"key": name, "file": f, "mtime": time.strftime("%Y-%m-%d %H:%M", time.localtime(mtime))})
    return models


def train_target(target_dir):
    base = os.path.abspath(param["base_directory"])
    rel = os.path.relpath(target_dir, base).replace(os.sep, '/')
    parts = rel.split('/')
    if len(parts) >= 3:
        db = parts[-3]
        machine_type = parts[-2]
        machine_id = parts[-1]
    elif len(parts) == 2:
        machine_id = parts[-1]
        combined = parts[-2]
        machine_type = combined.split('_')[-1]
        db = '_'.join(combined.split('_')[:-1]) if '_' in combined else combined
    else:
        machine_id = parts[-1]
        machine_type = machine_id
        db = "unknown"
    key = f"{machine_type}_{machine_id}_{db}"

    state["current"] = key
    state["progress"].append({"key": key, "status": "processing", "auc": None})

    train_pickle = f"{param['pickle_directory']}/train_{key}.pickle"
    eval_files_pickle = f"{param['pickle_directory']}/eval_files_{key}.pickle"
    eval_labels_pickle = f"{param['pickle_directory']}/eval_labels_{key}.pickle"
    eval_patches_pickle = f"{param['pickle_directory']}/eval_patches_{key}.pickle"
    scaler_pickle = f"{param['pickle_directory']}/scaler_{key}.pickle"
    model_file = f"{param['model_directory']}/model_{key}.pth"

    os.makedirs(param["pickle_directory"], exist_ok=True)
    os.makedirs(param["model_directory"], exist_ok=True)

    model_type = param["fit"].get("model_type", "ae")
    is_cnn = model_type == "cnn"
    is_vae = model_type == "vae"

    # --- Dataset ---
    log(f"[{key}] Loading dataset...")
    need_extract = not (os.path.exists(train_pickle) and os.path.exists(eval_files_pickle) and os.path.exists(eval_labels_pickle))
    if need_extract:
        train_files, _, eval_files, eval_labels = dataset_generator(target_dir)
        if is_cnn:
            train_data = list_to_patch_array(
                train_files,
                msg=f"[{key}] patches",
                n_mels=param["feature"]["n_mels"],
                patch_frames=param["feature"].get("patch_frames", 64),
                stride=param["feature"].get("patch_stride", 32),
                n_fft=param["feature"]["n_fft"],
                hop_length=param["feature"]["hop_length"],
                power=param["feature"]["power"],
            )
        else:
            train_data = list_to_vector_array(
                train_files,
                msg=f"[{key}] feat",
                n_mels=param["feature"]["n_mels"],
                frames=param["feature"]["frames"],
                n_fft=param["feature"]["n_fft"],
                hop_length=param["feature"]["hop_length"],
                power=param["feature"]["power"],
            )
        save_pickle(train_pickle, train_data)
        save_pickle(eval_files_pickle, eval_files)
        save_pickle(eval_labels_pickle, eval_labels)
        log(f"[{key}] Created: {len(train_data)} train samples")
    else:
        train_data = load_pickle(train_pickle)
        eval_files = load_pickle(eval_files_pickle)
        eval_labels = load_pickle(eval_labels_pickle)
        log(f"[{key}] Cached: {len(train_data)} train samples")

    # Pre-cache eval patches for faster evaluation
    if is_cnn and not os.path.exists(eval_patches_pickle):
        log(f"[{key}] Pre-caching eval patches...")
        eval_patches = {}
        for fn in eval_files:
            p = file_to_patches(fn,
                n_mels=param["feature"]["n_mels"],
                patch_frames=param["feature"].get("patch_frames", 64),
                stride=param["feature"].get("patch_stride", 32),
                n_fft=param["feature"]["n_fft"],
                hop_length=param["feature"]["hop_length"],
                power=param["feature"]["power"],
            )
            eval_patches[fn] = p
        save_pickle(eval_patches_pickle, eval_patches)
        log(f"[{key}] Eval patches cached ({len(eval_patches)} files)")
    elif is_cnn:
        eval_patches = load_pickle(eval_patches_pickle)
    else:
        eval_patches = None
        train_files, _, eval_files, eval_labels = dataset_generator(target_dir)
        if is_cnn:
            train_data = list_to_patch_array(
                train_files,
                msg=f"[{key}] patches",
                n_mels=param["feature"]["n_mels"],
                patch_frames=param["feature"].get("patch_frames", 64),
                stride=param["feature"].get("patch_stride", 32),
                n_fft=param["feature"]["n_fft"],
                hop_length=param["feature"]["hop_length"],
                power=param["feature"]["power"],
            )
        else:
            train_data = list_to_vector_array(
                train_files,
                msg=f"[{key}] feat",
                n_mels=param["feature"]["n_mels"],
                frames=param["feature"]["frames"],
                n_fft=param["feature"]["n_fft"],
                hop_length=param["feature"]["hop_length"],
                power=param["feature"]["power"],
            )
        save_pickle(train_pickle, train_data)
        save_pickle(eval_files_pickle, eval_files)
        save_pickle(eval_labels_pickle, eval_labels)
        log(f"[{key}] Created: {len(train_data)} train samples")

    if len(train_data) == 0:
        log(f"[{key}] SKIP: empty data")
        for p in state["progress"]:
            if p["key"] == key:
                p["status"] = "error"
        state["current"] = None
        return

    # --- Model ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if is_cnn:
        cnn_base_dim = param["feature"].get("cnn_base_dim", 16)
        model = CNNAutoencoder(in_channels=1, base_dim=cnn_base_dim).to(device)
        input_dim = None
        bottleneck_size = None
    else:
        input_dim = param["feature"]["n_mels"] * param["feature"]["frames"]
        bottleneck_size = param["fit"].get("bottleneck_size", 16)
        if model_type == "vae":
            model = VAE(input_dim, bottleneck_size=bottleneck_size).to(device)
        else:
            model = Autoencoder(input_dim, bottleneck_size=bottleneck_size).to(device)

    use_norm = param["fit"].get("normalize", False)
    load_ok = False
    if os.path.exists(model_file) and (not use_norm or os.path.exists(scaler_pickle)):
        try:
            model.load_state_dict(torch.load(model_file, map_location=device, weights_only=True))
            scaler = load_pickle(scaler_pickle) if use_norm else None
            log(f"[{key}] Model loaded from cache")
            load_ok = True
        except Exception as e:
            log(f"[{key}] Model cache mismatch ({e}), retraining from scratch...")
            try:
                os.remove(model_file)
            except:
                pass

    if not load_ok:
        if use_norm and not is_cnn:
            scaler = StandardScaler()
            train_data = scaler.fit_transform(train_data)
            save_pickle(scaler_pickle, scaler)
        else:
            scaler = None

        log(f"[{key}] Training ({param['fit']['epochs']} epochs, model={model_type})...")
        beta = param["fit"].get("beta", 0.1)
        denoising_std = param["fit"].get("denoising_std", 0.0)
        train_tensor = torch.from_numpy(train_data).float()
        val_split = param["fit"].get("validation_split", 0.1)
        val_len = int(len(train_tensor) * val_split)
        train_len = len(train_tensor) - val_len
        train_t, val_t = random_split(train_tensor, [train_len, val_len])
        train_loader = DataLoader(train_t, batch_size=param["fit"]["batch_size"], shuffle=True)
        val_loader = DataLoader(val_t, batch_size=param["fit"]["batch_size"], shuffle=False)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=param["fit"].get("learning_rate", 0.001))
        epochs = param["fit"]["epochs"]
        log_interval = max(1, epochs // 10)

        for epoch in range(epochs):
            if not state["training"]:
                log(f"[{key}] Training stopped")
                return
            model.train()
            total_loss = 0
            total_grad_norm = 0
            n_batches = 0
            for batch_x in train_loader:
                bx = batch_x.to(device)
                if denoising_std > 0:
                    noisy_x = bx + torch.randn_like(bx) * denoising_std
                else:
                    noisy_x = bx
                if is_vae:
                    recon, mu, log_var = model(noisy_x)
                    recon_loss = criterion(recon, bx)
                    kl = model.kl_loss(mu, log_var).mean()
                    loss = recon_loss + beta * kl
                else:
                    recon = model(noisy_x)
                    loss = criterion(recon, bx)
                optimizer.zero_grad()
                loss.backward()
                gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1e6)
                optimizer.step()
                total_loss += loss.item() * bx.size(0)
                total_grad_norm += gn.item()
                n_batches += 1
            train_loss = total_loss / len(train_loader.dataset)
            grad_norm = total_grad_norm / n_batches

            model.eval()
            total_val = 0
            with torch.no_grad():
                for batch_x in val_loader:
                    bx = batch_x.to(device)
                    if is_vae:
                        recon, mu, log_var = model(bx)
                        recon_loss = criterion(recon, bx)
                        kl = model.kl_loss(mu, log_var).mean()
                        loss = recon_loss + beta * kl
                    else:
                        recon = model(bx)
                        loss = criterion(recon, bx)
                    total_val += loss.item() * bx.size(0)
            val_loss = total_val / len(val_loader.dataset)

            if epoch == 0:
                log(f"[{key}]    Epoch     loss   val_loss     grad_norm")
            if (epoch + 1) % log_interval == 0 or epoch == 0:
                log(f"[{key}] {epoch+1:5d}/{epochs}  {train_loss:.4f}  {val_loss:.4f}  {grad_norm:.4f}")

        log(f"[{key}] Training done ({epochs} epochs, bottleneck={bottleneck_size}, denoising={denoising_std}, score_mode={param['fit'].get('score_mode','mean')})")
        torch.save(model.state_dict(), model_file)

    # --- Evaluation ---
    log(f"[{key}] Evaluating {len(eval_files)} files...")
    model.eval()
    score_mode = param["fit"].get("score_mode", "mean")
    y_pred = np.zeros(len(eval_labels))
    all_frame_errors = []
    eval_log_interval = max(1, len(eval_files) // 10)
    for i, fn in enumerate(eval_files):
        if not state["training"]:
            return
        try:
            if is_cnn:
                cached = eval_patches.get(fn) if eval_patches else None
                data = cached if cached is not None else file_to_patches(fn,
                    n_mels=param["feature"]["n_mels"],
                    patch_frames=param["feature"].get("patch_frames", 64),
                    stride=param["feature"].get("patch_stride", 32),
                    n_fft=param["feature"]["n_fft"],
                    hop_length=param["feature"]["hop_length"],
                    power=param["feature"]["power"],
                )
            else:
                data = file_to_vector_array(
                    fn,
                    n_mels=param["feature"]["n_mels"],
                    frames=param["feature"]["frames"],
                    n_fft=param["feature"]["n_fft"],
                    hop_length=param["feature"]["hop_length"],
                    power=param["feature"]["power"],
                )
            if data.shape[0] == 0:
                y_pred[i] = 0.0
                all_frame_errors.append(np.array([0.0]))
                continue
            if not is_cnn and scaler is not None:
                data = scaler.transform(data)
            dt = torch.from_numpy(data).float().to(device)
            with torch.no_grad():
                out = model(dt)
                if isinstance(out, tuple):
                    recon = out[0]
                else:
                    recon = out
                if is_cnn:
                    frame_errors = torch.mean((dt - recon) ** 2, dim=[1, 2, 3]).cpu().numpy()
                else:
                    frame_errors = torch.mean((dt - recon) ** 2, dim=1).cpu().numpy()
            all_frame_errors.append(frame_errors)
            if score_mode == "max":
                y_pred[i] = float(np.max(frame_errors))
            elif score_mode == "p95":
                y_pred[i] = float(np.percentile(frame_errors, 95))
            else:
                y_pred[i] = float(np.mean(frame_errors))
            if (i + 1) % eval_log_interval == 0:
                label = "N" if eval_labels[i] == 0 else "A"
                log(f"[{key}]   [{i+1}/{len(eval_files)}] file={os.path.basename(fn)} label={label} score={y_pred[i]:.4f}")
        except Exception as e:
            log(f"[{key}] Eval error: {e}")
            y_pred[i] = 0.0
            all_frame_errors.append(np.array([0.0]))

    auc = metrics.roc_auc_score(eval_labels, y_pred)
    log(f"[{key}] AUC = {auc:.6f}")

    # ── Detailed stats ──
    y_pred_arr = np.array(y_pred)
    labels_arr = np.array(eval_labels)
    normal_scores = y_pred_arr[labels_arr == 0]
    anomal_scores = y_pred_arr[labels_arr == 1]

    def score_stats(arr):
        return {
            "mean": float(np.mean(arr)), "std": float(np.std(arr)),
            "min": float(np.min(arr)), "max": float(np.max(arr)),
            "median": float(np.median(arr)),
            "p5": float(np.percentile(arr, 5)),
            "p95": float(np.percentile(arr, 95)),
            "count": int(len(arr)),
        }

    stats = {
        "AUC": float(auc),
        "normal_scores": score_stats(normal_scores),
        "anomal_scores": score_stats(anomal_scores),
    }

    # EER & best threshold via Youden index
    fpr, tpr, ths = metrics.roc_curve(labels_arr, y_pred_arr)
    fnr = 1 - tpr
    eer_idx = np.nanargmin(np.abs(fpr - fnr))
    stats["EER"] = float(fpr[eer_idx])
    stats["EER_threshold"] = float(ths[eer_idx])
    youden = tpr - fpr
    best_i = int(np.argmax(youden))
    stats["best_threshold"] = float(ths[best_i])
    stats["best_Youden"] = float(youden[best_i])
    pred_best = (y_pred_arr > ths[best_i]).astype(int)
    stats["best_F1"] = float(metrics.f1_score(labels_arr, pred_best))
    stats["best_precision"] = float(metrics.precision_score(labels_arr, pred_best, zero_division=0))
    stats["best_recall"] = float(metrics.recall_score(labels_arr, pred_best, zero_division=0))

    # Frame-level stats per file
    frame_means = np.array([np.mean(fe) for fe in all_frame_errors])
    frame_maxs = np.array([np.max(fe) for fe in all_frame_errors])
    stats["frame_mean_avg"] = float(np.mean(frame_means))
    stats["frame_mean_std"] = float(np.std(frame_means))
    stats["frame_max_avg"] = float(np.mean(frame_maxs))
    stats["frame_max_std"] = float(np.std(frame_maxs))

    # Score histogram (20 bins) for normal vs anomal
    all_scores = np.concatenate([normal_scores, anomal_scores])
    bin_edges = np.linspace(all_scores.min(), all_scores.max(), 21)
    n_hist, _ = np.histogram(normal_scores, bins=bin_edges)
    a_hist, _ = np.histogram(anomal_scores, bins=bin_edges)
    stats["histogram"] = {
        "bin_edges": bin_edges.tolist(),
        "normal": n_hist.tolist(),
        "anomal": a_hist.tolist(),
    }

    # Config snapshot
    stats["config"] = {
        "model": model_type,
        "bottleneck": bottleneck_size if not is_cnn else "-",
        "beta": beta if is_vae else "-",
        "denoising_std": denoising_std,
        "epochs": epochs,
        "score_mode": score_mode,
        "normalize": use_norm,
        "n_mels": param["feature"]["n_mels"],
        "frames": param["feature"].get("patch_frames", 64) if is_cnn else param["feature"]["frames"],
    }

    log(f"[{key}] Normal: mean={normal_scores.mean():.4f}±{normal_scores.std():.4f}, "
        f"Anomal: mean={anomal_scores.mean():.4f}±{anomal_scores.std():.4f}")
    log(f"[{key}] EER={stats['EER']:.4f}, best_th={stats['best_threshold']:.4f}, best_F1={stats['best_F1']:.4f}")

    for p in state["progress"]:
        if p["key"] == key:
            p["status"] = "done"
            p["auc"] = float(auc)
    state["results"][key] = stats

    # Save to result file
    result_file = f"{param['result_directory']}/{param['result_file']}"
    os.makedirs(param["result_directory"], exist_ok=True)
    saved = {}
    if os.path.exists(result_file):
        with open(result_file) as f:
            saved = yaml.safe_load(f) or {}
    saved.update(state["results"])
    with open(result_file, "w") as f:
        yaml.dump(saved, f)

    state["current"] = None


def log(msg):
    timestamp = time.strftime("%H:%M:%S")
    try:
        state["log_queue"].put_nowait(f"[{timestamp}] {msg}")
    except queue.Full:
        pass


def background_train(targets=None):
    state["training"] = True
    state["progress"] = []
    state["results"] = {}
    state["log_queue"] = queue.Queue(maxsize=2000)
    try:
        load_config()
        all_ds = get_available_datasets()
        if not all_ds:
            log("No datasets found in dataset directory")
            return
        if targets:
            all_ds = [d for d in all_ds if d["key"] in targets]
            if not all_ds:
                log("No matching datasets found")
                return
        for d in all_ds:
            if not state["training"]:
                break
            train_target(d["path"])
    except Exception as e:
        log(f"FATAL: {e}")
        import traceback
        for line in traceback.format_exc().splitlines():
            log(line)
    finally:
        state["training"] = False
        state["current"] = None
        log("Training finished")


# ── Routes ──────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/config", methods=["GET", "POST"])
def api_config():
    load_config()
    if request.method == "POST":
        updates = request.get_json(silent=True) or {}
        for section, keys in updates.items():
            if section in param and isinstance(param[section], dict):
                for k, v in keys.items():
                    if k in param[section]:
                        param[section][k] = v
        with open("baseline.yaml", "w") as f:
            yaml.dump(param, f, default_flow_style=False)
        return jsonify({"status": "saved"})
    return jsonify(param)


@app.route("/api/datasets")
def api_datasets():
    try:
        return jsonify(get_available_datasets())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/models")
def api_models():
    try:
        return jsonify(get_trained_models())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/train", methods=["POST"])
def api_train():
    if state["training"]:
        return jsonify({"error": "Training already in progress"}), 400
    data = request.get_json(silent=True) or {}
    targets = data.get("targets")
    thread = threading.Thread(target=background_train, args=(targets,), daemon=True)
    thread.start()
    return jsonify({"status": "started"})


@app.route("/api/stop", methods=["POST"])
def api_stop():
    state["training"] = False
    return jsonify({"status": "stopping"})


@app.route("/api/status")
def api_status():
    logs = []
    while not state["log_queue"].empty():
        try:
            logs.append(state["log_queue"].get_nowait())
        except queue.Empty:
            break
    return jsonify({
        "training": state["training"],
        "current": state["current"],
        "progress": state["progress"],
        "logs": logs,
        "results": state["results"],
    })


@app.route("/api/results")
def api_results():
    load_config()
    result_file = f"{param['result_directory']}/{param['result_file']}"
    saved = {}
    if os.path.exists(result_file):
        with open(result_file) as f:
            saved = yaml.safe_load(f) or {}
    merged = {**saved, **state["results"]}
    return jsonify(merged)


@app.route("/api/predict", methods=["POST"])
def api_predict():
    load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_type = param["fit"].get("model_type", "ae")
    is_cnn = model_type == "cnn"

    # Get audio
    if "file" in request.files:
        f = request.files["file"]
        data_bytes = f.read()
        import io
        audio_data, sr = librosa.load(io.BytesIO(data_bytes), sr=None, mono=False)
        model_key = request.form.get("model_key", "")
    else:
        body = request.get_json(silent=True) or {}
        filepath = body.get("path", "")
        if not filepath or not os.path.exists(filepath):
            return jsonify({"error": "File not found"}), 404
        audio_data, sr = librosa.load(filepath, sr=None, mono=False)
        model_key = body.get("model_key", "")

    if audio_data.ndim > 1:
        audio_data = audio_data[0, :]

    # Load model
    if not model_key:
        models = get_trained_models()
        if not models:
            return jsonify({"error": "No trained models found. Train a model first."}), 404
        model_key = models[0]["key"]

    model_path = f"{param['model_directory']}/model_{model_key}.pth"
    if not os.path.exists(model_path):
        return jsonify({"error": f"Model '{model_key}' not found"}), 404

    if is_cnn:
        cnn_base_dim = param["feature"].get("cnn_base_dim", 16)
        model = CNNAutoencoder(in_channels=1, base_dim=cnn_base_dim).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.eval()

        # Extract patches
        patch_data = audio_to_patches(
            audio_data, sr,
            n_mels=param["feature"]["n_mels"],
            patch_frames=param["feature"].get("patch_frames", 64),
            stride=param["feature"].get("patch_stride", 32),
            n_fft=param["feature"]["n_fft"],
            hop_length=param["feature"]["hop_length"],
            power=param["feature"]["power"],
        )
        if patch_data.shape[0] == 0:
            return jsonify({"error": "Audio too short"}), 400
        dt = torch.from_numpy(patch_data).float().to(device)
        with torch.no_grad():
            out = model(dt)
            if isinstance(out, tuple):
                recon = out[0]
            else:
                recon = out
            patch_errors = torch.mean((dt - recon) ** 2, dim=[1, 2, 3]).cpu().numpy()
        score = float(np.max(patch_errors))
        return jsonify({
            "score": score,
            "score_str": f"{score:.6f}",
            "model_key": model_key,
            "num_frames": len(patch_errors),
        })
    else:
        input_dim = param["feature"]["n_mels"] * param["feature"]["frames"]
        mel = librosa.feature.melspectrogram(
            y=audio_data, sr=sr,
            n_fft=param["feature"]["n_fft"],
            hop_length=param["feature"]["hop_length"],
            n_mels=param["feature"]["n_mels"],
            power=param["feature"]["power"],
        )
        log_mel = 20.0 / param["feature"]["power"] * np.log10(mel + sys.float_info.epsilon)
        vec_frames = param["feature"]["frames"]
        vec_size = log_mel.shape[1] - vec_frames + 1
        if vec_size < 1:
            return jsonify({"error": f"Audio too short ({log_mel.shape[1]} frames, need >= {vec_frames})"}), 400
        vec = np.zeros((vec_size, input_dim), float)
        for t in range(vec_frames):
            vec[:, param["feature"]["n_mels"] * t: param["feature"]["n_mels"] * (t + 1)] = log_mel[:, t: t + vec_size].T

        bottleneck_size = param["fit"].get("bottleneck_size", 16)
        if model_type == "vae":
            model = VAE(input_dim, bottleneck_size=bottleneck_size).to(device)
        else:
            model = Autoencoder(input_dim, bottleneck_size=bottleneck_size).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.eval()

        dt = torch.from_numpy(vec).float().to(device)
        with torch.no_grad():
            out = model(dt)
            if isinstance(out, tuple):
                recon = out[0]
            else:
                recon = out
            frame_errors = torch.mean((dt - recon) ** 2, dim=1).cpu().numpy()

        score = float(np.max(frame_errors))
        return jsonify({
            "score": score,
            "score_str": f"{score:.6f}",
            "model_key": model_key,
            "num_frames": len(frame_errors),
        })


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--public", action="store_true", help="Expose via ngrok on 0.0.0.0")
    args = parser.parse_args()

    load_config()
    port = int(os.environ.get("PORT", 228))
    host = "0.0.0.0" if args.public else "127.0.0.1"
    print(f"Starting MIMII GUI at http://{host}:{port}")

    if args.public:
        try:
            import ngrok
            listener = ngrok.forward(port, authtoken_from_env=True)
            public_url = listener.url()
            print("\n" + "=" * 52)
            print(f"  PUBLIC URL: {public_url}")
            print("=" * 52 + "\n")
        except Exception as e:
            print(f"WARNING: ngrok failed ({e}), continuing locally")

    app.run(host=host, port=port, debug=False, threaded=True)
