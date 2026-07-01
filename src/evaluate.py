"""Evaluation: AUC-ROC, pAUC, macro-F1, confusion matrix, SNR robustness."""

import numpy as np
import torch
from sklearn.metrics import (
    roc_auc_score, roc_curve, f1_score, accuracy_score,
    confusion_matrix, classification_report,
)
from torch.utils.data import DataLoader

from src.config import N_MELS


def partial_auc(y_true: np.ndarray, y_score: np.ndarray,
                max_fpr: float = 0.1) -> float:
    """Compute partial AUC up to max_fpr, normalized to [0, 1]."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    mask = fpr <= max_fpr
    if mask.sum() < 2:
        return 0.0
    fpr_clipped = fpr[mask]
    tpr_clipped = tpr[mask]
    _trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    pauc = _trapz(tpr_clipped, fpr_clipped)
    pauc_norm = pauc / max_fpr
    return pauc_norm


def evaluate_anomaly_detection(model, test_loader, test_labels: np.ndarray,
                                device: torch.device, threshold: float | None = None):
    """Evaluate anomaly detection performance."""
    model.eval()
    errors = []
    with torch.no_grad():
        for batch in test_loader:
            if isinstance(batch, (list, tuple)):
                x = batch[0]
            else:
                x = batch
            x = x.to(device)
            err = model.reconstruction_error(x)
            errors.extend(err.cpu().numpy())
    errors = np.array(errors)

    auc = roc_auc_score(test_labels, errors)
    pauc = partial_auc(test_labels, errors, max_fpr=0.1)

    results = {
        "auc_roc": auc,
        "pauc_01": pauc,
        "errors": errors,
    }

    if threshold is not None:
        preds = (errors > threshold).astype(int)
        results["accuracy"] = accuracy_score(test_labels, preds)
        results["f1"] = f1_score(test_labels, preds)
        results["confusion_matrix"] = confusion_matrix(test_labels, preds)

    print(f"  AUC-ROC: {auc:.4f}")
    print(f"  pAUC(FPR<=0.1): {pauc:.4f}")
    if threshold is not None:
        print(f"  Accuracy: {results['accuracy']:.4f}")
        print(f"  F1: {results['f1']:.4f}")
        print(f"  Confusion Matrix:\n{results['confusion_matrix']}")

    return results


def evaluate_classifier(classifier, features: np.ndarray, labels: np.ndarray,
                        device: torch.device, class_names: list[str] | None = None):
    """Evaluate defect type classifier."""
    classifier.eval()
    with torch.no_grad():
        logits = classifier(torch.from_numpy(features).float().to(device))
        preds = logits.argmax(dim=1).cpu().numpy()

    acc = accuracy_score(labels, preds)
    f1_macro = f1_score(labels, preds, average="macro", zero_division=0)
    cm = confusion_matrix(labels, preds)
    report = classification_report(labels, preds, target_names=class_names, zero_division=0)

    results = {
        "accuracy": acc,
        "macro_f1": f1_macro,
        "confusion_matrix": cm,
        "predictions": preds,
        "report": report,
    }

    print(f"  Accuracy: {acc:.4f}")
    print(f"  Macro-F1: {f1_macro:.4f}")
    print(f"  Report:\n{report}")

    return results


def evaluate_snr_robustness(model, machine_type: str, snr_levels: list[int],
                            device: torch.device, threshold: float,
                            batch_size: int = 64,
                            norm_stats: tuple[float, float] | None = None):
    """Evaluate anomaly detection across different SNR levels.

    `norm_stats` should be the SAME (min, max) dB stats the model was
    trained with, so scores stay comparable across SNR levels.
    """
    from src.dataset import build_datasets

    results = {}
    for snr in snr_levels:
        print(f"\n--- SNR = {snr} dB ---")
        try:
            _, _, test_ds, test_labels, _, _ = build_datasets(
                machine_type, snr_db=snr, augment_train=False, norm_stats=norm_stats,
            )
            test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
            res = evaluate_anomaly_detection(model, test_loader, test_labels,
                                             device, threshold)
            results[snr] = res
        except FileNotFoundError:
            print(f"  Data for SNR={snr}dB not found, skipping.")
    return results
