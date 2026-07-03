#!/usr/bin/env python
"""Diagnostic: check data, features, and model predictions for id_00."""
import os, sys, glob
import numpy as np
import librosa
import yaml
import logging
logging.basicConfig(level=logging.WARNING)

from baseline import (
    dataset_generator, file_to_vector_array, list_to_vector_array,
    load_pickle, save_pickle, Autoencoder,
)
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split

with open("baseline.yaml") as f:
    param = yaml.safe_load(f)

target_dir = os.path.abspath("dataset/-6_dB_fan/id_00")
print(f"Target: {target_dir}")
print(f"Exists: {os.path.exists(target_dir)}")

# 1. Check file counts
normal = sorted(glob.glob(f"{target_dir}/normal/*.wav"))
abnormal = sorted(glob.glob(f"{target_dir}/abnormal/*.wav"))
print(f"\nNormal files:   {len(normal)}")
print(f"Abnormal files: {len(abnormal)}")

if len(normal) > 0:
    from baseline import demux_wav
    sr, y = demux_wav(normal[0])
    print(f"Sample rate: {sr}, samples: {len(y)}, duration: {len(y)/sr:.2f}s")
    print(f"Channels: {y.ndim if hasattr(y,'ndim') else 1}")

# 2. Check feature extraction on one file
print("\n--- Feature extraction (one normal file) ---")
feat = file_to_vector_array(
    normal[0], n_mels=param["feature"]["n_mels"],
    frames=param["feature"]["frames"],
    n_fft=param["feature"]["n_fft"],
    hop_length=param["feature"]["hop_length"],
    power=param["feature"]["power"],
)
print(f"Shape: {feat.shape}")
print(f"Range: [{feat.min():.4f}, {feat.max():.4f}]")
print(f"Mean:  {feat.mean():.4f}")
print(f"Std:   {feat.std():.4f}")
print(f"NaN count: {np.isnan(feat).sum()}")

# 3. Train a quick model and check
print("\n--- Quick training ---")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

train_files, _, eval_files, eval_labels = dataset_generator(target_dir)
print(f"Train files: {len(train_files)}, Eval files: {len(eval_files)}")

train_data = list_to_vector_array(
    train_files, msg="feat",
    n_mels=param["feature"]["n_mels"],
    frames=param["feature"]["frames"],
    n_fft=param["feature"]["n_fft"],
    hop_length=param["feature"]["hop_length"],
    power=param["feature"]["power"],
)
print(f"Train data shape: {train_data.shape}")
print(f"Train data range: [{train_data.min():.4f}, {train_data.max():.4f}]")
print(f"NaN in train: {np.isnan(train_data).sum()}")

input_dim = param["feature"]["n_mels"] * param["feature"]["frames"]
model = Autoencoder(input_dim, bottleneck_size=8).to(device)

dataset = TensorDataset(torch.from_numpy(train_data).float())
val_split = 0.1
val_len = int(len(dataset) * val_split)
train_len = len(dataset) - val_len
train_ds, val_ds = random_split(dataset, [train_len, val_len])
train_loader = DataLoader(train_ds, batch_size=512, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=512, shuffle=False)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print("\nEpoch  loss    val_loss")
for epoch in range(50):
    model.train()
    tl = 0
    for bx in train_loader:
        bx = bx[0].to(device)
        recon = model(bx)
        loss = criterion(recon, bx)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        tl += loss.item() * bx.size(0)
    train_loss = tl / len(train_loader.dataset)

    model.eval()
    vl = 0
    with torch.no_grad():
        for bx in val_loader:
            bx = bx[0].to(device)
            recon = model(bx)
            loss = criterion(recon, bx)
            vl += loss.item() * bx.size(0)
    val_loss = vl / len(val_loader.dataset)

    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"{epoch+1:3d}   {train_loss:.6f}  {val_loss:.6f}")

# 4. Evaluate
print("\n--- Evaluation ---")
model.eval()
y_pred = np.zeros(len(eval_labels))
for i, fn in enumerate(eval_files):
    data = file_to_vector_array(
        fn, n_mels=param["feature"]["n_mels"],
        frames=param["feature"]["frames"],
        n_fft=param["feature"]["n_fft"],
        hop_length=param["feature"]["hop_length"],
        power=param["feature"]["power"],
    )
    if data.shape[0] == 0:
        y_pred[i] = 0.0
        continue
    dt = torch.from_numpy(data).float().to(device)
    with torch.no_grad():
        recon = model(dt)
        err = torch.mean((dt - recon) ** 2, dim=1).cpu().numpy()
    y_pred[i] = np.mean(err)

# 5. Score distribution
normal_scores = y_pred[:len(eval_labels)//2]
anomal_scores = y_pred[len(eval_labels)//2:]
print(f"\nNormal scores: mean={normal_scores.mean():.6f} std={normal_scores.std():.6f}")
print(f"  min={normal_scores.min():.6f} max={normal_scores.max():.6f}")
print(f"Anomal scores: mean={anomal_scores.mean():.6f} std={anomal_scores.std():.6f}")
print(f"  min={anomal_scores.min():.6f} max={anomal_scores.max():.6f}")

from sklearn import metrics
auc = metrics.roc_auc_score(eval_labels, y_pred)
print(f"\nAUC = {auc:.6f}")
