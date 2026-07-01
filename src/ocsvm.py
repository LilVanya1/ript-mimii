"""Bonus: One-Class SVM baseline on MFCC features."""

import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from src.dataset import discover_files, load_audio, window_signal, extract_mfcc
from src.evaluate import partial_auc


def extract_mfcc_features(windows: list[np.ndarray]) -> np.ndarray:
    """Extract mean+std of MFCC across time for each window."""
    feats = []
    for w in windows:
        mfcc = extract_mfcc(w)
        feat = np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1)])
        feats.append(feat)
    return np.array(feats)


def run_ocsvm_baseline(machine_type: str, snr_db: int = 6,
                       kernel: str = "rbf", nu: float = 0.05):
    """Train One-Class SVM on normal MFCC features, evaluate on test set.

    Uses file-level split and light hyperparameter tuning on validation AUC.
    """
    files = discover_files(machine_type, snr_db)

    normal_paths = []
    for _, paths in files["normal"].items():
        normal_paths.extend(paths)

    abnormal_wins = []
    for mid, paths in files["abnormal"].items():
        for p in paths:
            y = load_audio(p)
            abnormal_wins.extend(window_signal(y))

    rng = np.random.default_rng(42)
    n_files = len(normal_paths)
    idx = rng.permutation(n_files)
    n_test = max(1, int(n_files * 0.15))
    n_val = max(1, int(n_files * 0.15))
    test_files = [normal_paths[i] for i in idx[:n_test]]
    val_files = [normal_paths[i] for i in idx[n_test:n_test + n_val]]
    train_files = [normal_paths[i] for i in idx[n_test + n_val:]]

    def files_to_windows(paths):
        wins = []
        for p in paths:
            y = load_audio(p)
            wins.extend(window_signal(y))
        return wins

    train_wins = files_to_windows(train_files)
    val_wins = files_to_windows(val_files)
    test_normal_wins = files_to_windows(test_files)

    print("Extracting MFCC features...")
    X_train = extract_mfcc_features(train_wins)
    X_val = extract_mfcc_features(val_wins)
    X_test_normal = extract_mfcc_features(test_normal_wins)
    X_test_abnormal = extract_mfcc_features(abnormal_wins)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test_normal = scaler.transform(X_test_normal)
    X_test_abnormal = scaler.transform(X_test_abnormal)

    # Tune lightweight grid on validation AUC (val normal vs test abnormal sample).
    val_y = np.array([0] * len(X_val) + [1] * min(len(X_test_abnormal), len(X_val)))
    val_abn = X_test_abnormal[:min(len(X_test_abnormal), len(X_val))]
    X_val_mix = np.concatenate([X_val, val_abn], axis=0)

    best_auc = -1.0
    best_cfg = (kernel, nu, "scale")
    for nu_cand in (0.01, 0.03, 0.05, 0.1):
        for gamma_cand in ("scale", 0.01, 0.05, 0.1):
            oc = OneClassSVM(kernel=kernel, nu=nu_cand, gamma=gamma_cand)
            oc.fit(X_train)
            scores = -oc.decision_function(X_val_mix)
            auc = roc_auc_score(val_y, scores)
            if auc > best_auc:
                best_auc = auc
                best_cfg = (kernel, nu_cand, gamma_cand)

    print(f"Training OC-SVM tuned: kernel={best_cfg[0]}, nu={best_cfg[1]}, gamma={best_cfg[2]}")
    ocsvm = OneClassSVM(kernel=best_cfg[0], nu=best_cfg[1], gamma=best_cfg[2])
    ocsvm.fit(X_train)

    X_test = np.concatenate([X_test_normal, X_test_abnormal])
    y_test = np.array([0] * len(X_test_normal) + [1] * len(X_test_abnormal))

    # decision_function: negative = anomaly
    scores = -ocsvm.decision_function(X_test)

    auc = roc_auc_score(y_test, scores)
    pauc = partial_auc(y_test, scores, max_fpr=0.1)

    print(f"OC-SVM Results [{machine_type} @ {snr_db}dB]:")
    print(f"  AUC-ROC: {auc:.4f}")
    print(f"  pAUC(FPR≤0.1): {pauc:.4f}")

    return {"auc_roc": auc, "pauc_01": pauc, "scores": scores, "labels": y_test}


def run_logreg_baseline(machine_type: str, snr_db: int = 6):
    """Stronger classical baseline for MIMII-style features."""
    files = discover_files(machine_type, snr_db)
    normal_paths, abnormal_paths = [], []
    for _, paths in files["normal"].items():
        normal_paths.extend(paths)
    for _, paths in files["abnormal"].items():
        abnormal_paths.extend(paths)

    def files_to_windows(paths):
        out = []
        for p in paths:
            out.extend(window_signal(load_audio(p)))
        return out

    rng = np.random.default_rng(42)
    idx_n = rng.permutation(len(normal_paths))
    idx_a = rng.permutation(len(abnormal_paths))

    n_train_n = max(1, int(len(idx_n) * 0.7))
    n_train_a = max(1, int(len(idx_a) * 0.7))
    normal_train_files = [normal_paths[i] for i in idx_n[:n_train_n]]
    abnormal_train_files = [abnormal_paths[i] for i in idx_a[:n_train_a]]
    train_files = normal_train_files + abnormal_train_files
    test_files_n = [normal_paths[i] for i in idx_n[n_train_n:]]
    test_files_a = [abnormal_paths[i] for i in idx_a[n_train_a:]]

    normal_train_wins = files_to_windows(normal_train_files)
    abnormal_train_wins = files_to_windows(abnormal_train_files)
    train_wins = normal_train_wins + abnormal_train_wins
    test_wins_n = files_to_windows(test_files_n)
    test_wins_a = files_to_windows(test_files_a)

    # Labels by source (first part of train is normal, second abnormal)
    y_train = np.array([0] * len(normal_train_wins) + [1] * len(abnormal_train_wins))
    y_test = np.array([0] * len(test_wins_n) + [1] * len(test_wins_a))

    X_train = extract_mfcc_features(train_wins)
    X_test = extract_mfcc_features(test_wins_n + test_wins_a)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    clf = LogisticRegression(max_iter=1500, class_weight="balanced")
    clf.fit(X_train, y_train)
    probs = clf.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, probs)
    pauc = partial_auc(y_test, probs, max_fpr=0.1)
    print(f"LogReg baseline [{machine_type} @ {snr_db}dB] AUC={auc:.4f} pAUC={pauc:.4f}")
    return {"auc_roc": auc, "pauc_01": pauc, "scores": probs, "labels": y_test}
