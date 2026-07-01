"""Defect type classifier on autoencoder bottleneck features."""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset

from src.config import LATENT_DIM


class DefectClassifier(nn.Module):
    """MLP classifier on latent features."""

    def __init__(self, latent_dim: int = LATENT_DIM, num_classes: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


def extract_latent_features(autoencoder, loader, device: torch.device):
    """Extract bottleneck features and labels from a DataLoader."""
    autoencoder.eval()
    features = []
    labels = []
    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                x, y = batch
                labels.extend(y.numpy())
            else:
                x = batch[0] if isinstance(batch, (list, tuple)) else batch
            x = x.to(device)
            z = autoencoder.encode(x)
            features.append(z.cpu().numpy())
    features = np.concatenate(features, axis=0)
    labels = np.array(labels) if labels else None
    return features, labels


def train_classifier(classifier: DefectClassifier, train_features: np.ndarray,
                     train_labels: np.ndarray, val_features: np.ndarray | None = None,
                     val_labels: np.ndarray | None = None, epochs: int = 50,
                     lr: float = 1e-3, batch_size: int = 64,
                     device: torch.device = torch.device("cpu"),
                     save_path: Path | None = None):
    """Train the defect classifier."""
    classifier = classifier.to(device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    train_ds = TensorDataset(
        torch.from_numpy(train_features).float(),
        torch.from_numpy(train_labels).long(),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    val_loader = None
    if val_features is not None and val_labels is not None:
        val_ds = TensorDataset(
            torch.from_numpy(val_features).float(),
            torch.from_numpy(val_labels).long(),
        )
        val_loader = DataLoader(val_ds, batch_size=batch_size)

    best_val_acc = 0.0
    history = {"train_loss": [], "val_acc": []}

    for epoch in range(1, epochs + 1):
        classifier.train()
        losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = classifier(xb)
            loss = criterion(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        t_loss = np.mean(losses)
        history["train_loss"].append(t_loss)

        if val_loader:
            classifier.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    preds = classifier(xb).argmax(dim=1)
                    correct += (preds == yb).sum().item()
                    total += len(yb)
            val_acc = correct / total
            history["val_acc"].append(val_acc)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                if save_path:
                    save_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(classifier.state_dict(), save_path)

            if epoch % 10 == 0:
                print(f"Epoch {epoch:3d} | loss={t_loss:.4f} | val_acc={val_acc:.3f}")
        else:
            if epoch % 10 == 0:
                print(f"Epoch {epoch:3d} | loss={t_loss:.4f}")

    if save_path and save_path.exists():
        classifier.load_state_dict(torch.load(save_path, weights_only=True))

    return classifier, history


def finetune_encoder_with_classifier(autoencoder: nn.Module, classifier: DefectClassifier,
                                     loader, epochs: int = 10, lr: float = 1e-4,
                                     device: torch.device = torch.device("cpu")):
    """Joint end-to-end fine-tuning of encoder+classifier on labeled batches."""
    autoencoder = autoencoder.to(device)
    classifier = classifier.to(device)
    autoencoder.train()
    classifier.train()

    optimizer = torch.optim.AdamW(
        list(autoencoder.encoder.parameters()) + list(classifier.parameters()),
        lr=lr, weight_decay=1e-4,
    )
    criterion = nn.CrossEntropyLoss()
    hist = {"loss": []}

    for epoch in range(1, epochs + 1):
        losses = []
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            z = autoencoder.encode(xb)
            logits = classifier(z)
            loss = criterion(logits, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        hist["loss"].append(float(np.mean(losses)))
        if epoch % 5 == 0:
            print(f"[joint-ft] epoch={epoch} loss={hist['loss'][-1]:.4f}")

    return autoencoder, classifier, hist
