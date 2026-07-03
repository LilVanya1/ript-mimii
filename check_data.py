#!/usr/bin/env python
"""Check if normal and abnormal sounds are separable at all."""
import os, sys, glob
import numpy as np
import librosa
import yaml
from baseline import file_to_vector_array, list_to_vector_array

with open("baseline.yaml") as f:
    param = yaml.safe_load(f)

target_dir = os.path.abspath("dataset/-6_dB_fan/id_00")
normal = sorted(glob.glob(f"{target_dir}/normal/*.wav"))
abnormal = sorted(glob.glob(f"{target_dir}/abnormal/*.wav"))
print(f"Normal: {len(normal)}, Abnormal: {len(abnormal)}")

# Check a few files
for label, files in [("normal", normal[:3]), ("abnormal", abnormal[:3])]:
    for f in files:
        try:
            y, sr = librosa.load(f, sr=None, mono=True)
            print(f"\n{label}: {os.path.basename(f)}")
            print(f"  duration={len(y)/sr:.2f}s sr={sr}")
            print(f"  rms={np.sqrt(np.mean(y**2)):.6f} peak={np.max(np.abs(y)):.6f}")
        except Exception as e:
            print(f"  ERROR: {e}")

# Extract features for all files (sample 100 each for speed)
print("\n\n--- Feature comparison (100 files each) ---")
n = min(100, len(normal), len(abnormal))

normal_feats = []
for f in normal[:n]:
    v = file_to_vector_array(f, n_mels=64, frames=5)
    if v.shape[0] > 0:
        normal_feats.append(v.mean(axis=0))
normal_feats = np.array(normal_feats)

abnormal_feats = []
for f in abnormal[:n]:
    v = file_to_vector_array(f, n_mels=64, frames=5)
    if v.shape[0] > 0:
        abnormal_feats.append(v.mean(axis=0))
abnormal_feats = np.array(abnormal_feats)

print(f"Normal mean vector:  mean={normal_feats.mean():.4f} std={normal_feats.mean(axis=0).std():.4f}")
print(f"Abnormal mean vector: mean={abnormal_feats.mean():.4f} std={abnormal_feats.mean(axis=0).std():.4f}")

# Simple classifier test
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

X = np.vstack([normal_feats, abnormal_feats])
y = np.array([0]*n + [1]*n)

clf = LogisticRegression(max_iter=1000, solver='lbfgs')
scores = cross_val_score(clf, X, y, cv=5, scoring='roc_auc')
print(f"\nLogisticRegression CV AUC: {scores.mean():.4f} ± {scores.std():.4f}")

# t-SNE visualization check
from sklearn.manifold import TSNE
try:
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    X_2d = tsne.fit_transform(X[:50])
    n_2d = X_2d[:50]
    a_2d = X_2d[50:]
    print(f"\nt-SNE (50 samples each):")
    print(f"  Normal cluster center: ({n_2d[:,0].mean():.2f}, {n_2d[:,1].mean():.2f})")
    print(f"  Anomal cluster center: ({a_2d[:,0].mean():.2f}, {a_2d[:,1].mean():.2f})")
    dist = np.sqrt((n_2d[:,0].mean() - a_2d[:,0].mean())**2 + (n_2d[:,1].mean() - a_2d[:,1].mean())**2)
    print(f"  Distance between centers: {dist:.2f}")
except Exception as e:
    print(f"  t-SNE error: {e}")
