"""Visualization utilities: reconstruction errors, t-SNE, heatmaps."""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import roc_curve
from pathlib import Path

from src.config import RESULTS_DIR


def plot_reconstruction_errors(normal_errors: np.ndarray, abnormal_errors: np.ndarray,
                                threshold: float, save_name: str = "reconstruction_errors.png"):
    """Histogram of reconstruction errors for normal vs abnormal."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(normal_errors, bins=80, alpha=0.6, label="Normal", color="steelblue", density=True)
    ax.hist(abnormal_errors, bins=80, alpha=0.6, label="Abnormal", color="tomato", density=True)
    ax.axvline(threshold, color="black", linestyle="--", linewidth=2, label=f"Threshold={threshold:.4f}")
    ax.set_xlabel("Reconstruction Error (MAE)")
    ax.set_ylabel("Density")
    ax.set_title("Distribution of Reconstruction Errors")
    ax.legend()
    plt.tight_layout()
    path = RESULTS_DIR / save_name
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")
    return path


def plot_roc_curve(y_true: np.ndarray, y_score: np.ndarray,
                   save_name: str = "roc_curve.png"):
    """Plot ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(y_true, y_score)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(fpr, tpr, linewidth=2, label=f"AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — Anomaly Detection")
    ax.legend(loc="lower right")
    plt.tight_layout()
    path = RESULTS_DIR / save_name
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")
    return path


def plot_tsne(features: np.ndarray, labels: np.ndarray,
              label_names: dict | None = None, save_name: str = "tsne.png",
              perplexity: int = 30):
    """t-SNE of latent space."""
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, n_iter=1000)
    emb = tsne.fit_transform(features)

    fig, ax = plt.subplots(figsize=(9, 7))
    unique = np.unique(labels)
    cmap = plt.cm.get_cmap("tab10", len(unique))
    for i, u in enumerate(unique):
        mask = labels == u
        name = label_names[u] if label_names else str(u)
        ax.scatter(emb[mask, 0], emb[mask, 1], c=[cmap(i)], label=name,
                   alpha=0.6, s=12, edgecolors="none")
    ax.set_title("t-SNE of Latent Space")
    ax.legend(markerscale=3)
    plt.tight_layout()
    path = RESULTS_DIR / save_name
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")
    return path


def plot_confusion_matrix(cm: np.ndarray, class_names: list[str],
                          save_name: str = "confusion_matrix.png"):
    """Plot confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix — Defect Classification")
    plt.tight_layout()
    path = RESULTS_DIR / save_name
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")
    return path


def plot_snr_comparison(snr_results: dict, save_name: str = "snr_comparison.png"):
    """Bar chart of AUC-ROC across SNR levels."""
    snrs = sorted(snr_results.keys())
    aucs = [snr_results[s]["auc_roc"] for s in snrs]
    paucs = [snr_results[s]["pauc_01"] for s in snrs]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(snrs))
    w = 0.35
    ax.bar(x - w / 2, aucs, w, label="AUC-ROC", color="steelblue")
    ax.bar(x + w / 2, paucs, w, label="pAUC(FPR≤0.1)", color="tomato")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s} dB" for s in snrs])
    ax.set_xlabel("SNR Level")
    ax.set_ylabel("Score")
    ax.set_title("Anomaly Detection Performance vs. SNR")
    ax.set_ylim(0, 1.05)
    ax.legend()
    for i, (a, p) in enumerate(zip(aucs, paucs)):
        ax.text(i - w / 2, a + 0.02, f"{a:.3f}", ha="center", fontsize=9)
        ax.text(i + w / 2, p + 0.02, f"{p:.3f}", ha="center", fontsize=9)
    plt.tight_layout()
    path = RESULTS_DIR / save_name
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")
    return path


def plot_spectrogram_comparison(original: np.ndarray, reconstructed: np.ndarray,
                                 save_name: str = "spectrogram_comparison.png"):
    """Side-by-side original vs reconstructed spectrograms."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].imshow(original, aspect="auto", origin="lower", cmap="magma")
    axes[0].set_title("Original")

    axes[1].imshow(reconstructed, aspect="auto", origin="lower", cmap="magma")
    axes[1].set_title("Reconstructed")

    diff = np.abs(original - reconstructed)
    axes[2].imshow(diff, aspect="auto", origin="lower", cmap="hot")
    axes[2].set_title("Absolute Difference")

    for ax in axes:
        ax.set_xlabel("Time Frame")
        ax.set_ylabel("Mel Band")
    plt.tight_layout()
    path = RESULTS_DIR / save_name
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")
    return path


def plot_training_history(history: dict, save_name: str = "training_history.png"):
    """Plot train/val loss curves."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(history["train_loss"], label="Train Loss")
    if "val_loss" in history:
        ax.plot(history["val_loss"], label="Val Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MAE Loss")
    ax.set_title("Autoencoder Training History")
    ax.legend()
    plt.tight_layout()
    path = RESULTS_DIR / save_name
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")
    return path
