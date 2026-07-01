"""Convolutional autoencoder for mel-spectrogram anomaly detection."""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import BASE_CHANNELS, LATENT_DIM


class ConvAutoencoder(nn.Module):
    """Convolutional autoencoder for mel-spectrogram anomaly detection.
    Trained with L1 reconstruction loss on normal data.
    Anomaly score = reconstruction error (higher = more anomalous).
    """

    def __init__(self, latent_dim: int = LATENT_DIM, base_channels: int = BASE_CHANNELS):
        super().__init__()
        c = max(8, int(base_channels))

        self.enc1 = self._conv_block(1, c)
        self.enc2 = self._conv_block(c, c * 2)
        self.enc3 = self._conv_block(c * 2, c * 4)
        self.enc4 = self._conv_block(c * 4, c * 8)

        self.dec4 = self._up_block(c * 8, c * 4)
        self.dec3 = self._up_block(c * 4, c * 2)
        self.dec2 = self._up_block(c * 2, c)
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(c, 1, 4, 2, 1),
            nn.Sigmoid(),
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Linear(c * 8, int(latent_dim))
        self.norm = nn.LayerNorm(int(latent_dim))

    @staticmethod
    def _conv_block(in_ch, out_ch, k=4, s=2, p=1):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, k, s, p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2),
        )

    @staticmethod
    def _up_block(in_ch, out_ch, k=4, s=2, p=1):
        return nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, k, s, p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2),
        )

    @staticmethod
    def _pad_time(x):
        T = x.shape[-1]
        pad = (16 - T % 16) % 16
        if pad > 0:
            x = F.pad(x, (0, pad))
        return x, pad

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Bottleneck features for defect classifier."""
        x, _ = self._pad_time(x)
        h = self.enc1(x)
        h = self.enc2(h)
        h = self.enc3(h)
        h = self.enc4(h)
        h = self.pool(h).flatten(1)
        z = self.proj(h)
        z = self.norm(z)
        return z

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Reconstruct input mel-spectrogram."""
        x, pad = self._pad_time(x)
        z = self.enc1(x)
        z = self.enc2(z)
        z = self.enc3(z)
        z = self.enc4(z)
        x_hat = self.dec4(z)
        x_hat = self.dec3(x_hat)
        x_hat = self.dec2(x_hat)
        x_hat = self.dec1(x_hat)
        if pad > 0:
            x_hat = x_hat[:, :, :, :-pad]
        return x_hat

    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """Per-sample L1 reconstruction error as anomaly score."""
        x_hat = self.forward(x)
        return torch.mean(torch.abs(x - x_hat), dim=(1, 2, 3))


def train_autoencoder(
    model: ConvAutoencoder,
    train_loader,
    val_loader,
    epochs: int,
    lr: float,
    patience: int,
    device: torch.device,
    save_path: Path | None = None,
    cb=None,
    use_amp: bool = False,
    stop_event=None,
):
    """Train ConvAutoencoder with L1 reconstruction loss on normal data."""
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=max(3, min(10, patience // 2))
    )
    loss_fn = nn.L1Loss()
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    max_grad_norm = 5.0

    best_val_loss = float("inf")
    wait = 0
    history = {"train_loss": [], "val_loss": [], "lr": []}

    for epoch in range(1, epochs + 1):
        if stop_event and stop_event.is_set():
            print(f"Training stopped by user at epoch {epoch}")
            stop_event.clear()
            if save_path and save_path.exists():
                model.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))
            break
        model.train()
        train_losses = []
        for batch in train_loader:
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            x = x.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            if use_amp:
                with torch.amp.autocast("cuda"):
                    x_hat = model(x)
                    loss = loss_fn(x_hat, x)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                x_hat = model(x)
                loss = loss_fn(x_hat, x)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
            train_losses.append(float(loss.item()))

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                x = batch[0] if isinstance(batch, (list, tuple)) else batch
                x = x.to(device, non_blocking=True)
                if use_amp:
                    with torch.amp.autocast("cuda"):
                        x_hat = model(x)
                        v_loss = loss_fn(x_hat, x)
                else:
                    x_hat = model(x)
                    v_loss = loss_fn(x_hat, x)
                val_losses.append(float(v_loss.item()))

        t_loss = float(np.mean(train_losses))
        v_loss = float(np.mean(val_losses))
        scheduler.step(v_loss)
        current_lr = float(optimizer.param_groups[0]["lr"])

        history["train_loss"].append(t_loss)
        history["val_loss"].append(v_loss)
        history["lr"].append(current_lr)

        if cb:
            cb(epoch, t_loss, v_loss)
        else:
            print(f"Epoch {epoch:3d} | train={t_loss:.6f} | val={v_loss:.6f} | lr={current_lr:.2e}")

        if v_loss < best_val_loss:
            best_val_loss = v_loss
            wait = 0
            if save_path:
                save_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), save_path)
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    if save_path and save_path.exists():
        model.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))

    return model, history


def _compute_scores(model: ConvAutoencoder, val_loader, device: torch.device) -> np.ndarray:
    model.eval()
    scores = []
    with torch.no_grad():
        for batch in val_loader:
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            x = x.to(device)
            s = model.reconstruction_error(x)
            scores.extend(s.cpu().numpy())
    return np.asarray(scores, dtype=np.float64)


def compute_threshold(
    model: ConvAutoencoder,
    val_loader,
    device: torch.device,
    quantile: float = 0.95,
    method: str = "kde_fpr",
    target_fpr: float = 0.05,
    mad_k: float = 3.0,
) -> float:
    """Compute anomaly score threshold from normal validation scores."""
    scores = _compute_scores(model, val_loader, device)
    if scores.size == 0:
        raise ValueError("No validation scores available for threshold estimation.")

    method = (method or "kde_fpr").lower()
    if method == "quantile":
        threshold = float(np.quantile(scores, quantile))
        print(f"Anomaly threshold [quantile q={quantile:.4f}]: {threshold:.6f}")
        return threshold

    if method == "mad":
        med = float(np.median(scores))
        mad = float(np.median(np.abs(scores - med)))
        mad_scaled = max(1e-12, 1.4826 * mad)
        threshold = med + mad_k * mad_scaled
        print(f"Anomaly threshold [mad k={mad_k:.3f}]: {threshold:.6f}")
        return float(threshold)

    if method != "kde_fpr":
        raise ValueError(f"Unknown threshold method: {method}")

    try:
        from scipy.stats import gaussian_kde

        if scores.size < 32 or not np.isfinite(scores).all():
            raise ValueError("Insufficient data for stable KDE")
        if float(np.std(scores)) < 1e-8:
            raise ValueError("Near-constant score distribution")

        kde = gaussian_kde(scores)
        lo = float(np.min(scores))
        hi = float(np.max(scores))
        span = max(1e-6, hi - lo)
        xs = np.linspace(lo - 0.2 * span, hi + 0.6 * span, 2000)
        pdf = kde(xs)
        dx = float(xs[1] - xs[0])
        cdf = np.cumsum(pdf) * dx
        cdf = cdf / max(1e-12, float(cdf[-1]))

        target_cdf = max(0.001, min(0.999, 1.0 - float(target_fpr)))
        idx = int(np.searchsorted(cdf, target_cdf))
        idx = max(0, min(idx, xs.size - 1))
        threshold = float(xs[idx])
        print(f"Anomaly threshold [kde_fpr fpr={target_fpr:.4f}]: {threshold:.6f}")
        return threshold
    except Exception as ex:
        med = float(np.median(scores))
        mad = float(np.median(np.abs(scores - med)))
        mad_scaled = max(1e-12, 1.4826 * mad)
        threshold = med + mad_k * mad_scaled
        print(f"Anomaly threshold [kde_fpr->mad fallback, reason={ex}]: {threshold:.6f}")
        return float(threshold)


# ── Baseline Dense Autoencoder (MIMII baseline reproduction) ─────────────

class DenseAutoencoder(nn.Module):
    """Simple dense autoencoder matching MIMII baseline architecture:
    320 -> 64 -> 64 -> 8 -> 64 -> 64 -> 320, MSE reconstruction.
    """
    def __init__(self, input_dim: int = 320):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 8),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(8, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.decoder(z)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        x_hat = self(x)
        err = torch.mean((x - x_hat) ** 2, dim=1)
        return err


def train_dense_autoencoder(
    model: DenseAutoencoder,
    train_loader,
    val_loader,
    epochs: int,
    lr: float,
    patience: int,
    device: torch.device,
    save_path: Path | None = None,
    cb=None,
):
    """Train DenseAutoencoder with MSE reconstruction (baseline style)."""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=max(3, patience // 2)
    )
    mse = nn.MSELoss()

    best_val_loss = float("inf")
    wait = 0
    history = {"train_loss": [], "val_loss": [], "lr": []}

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for batch in train_loader:
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            x = x.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            x_hat = model(x)
            loss = mse(x_hat, x)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                x = batch[0] if isinstance(batch, (list, tuple)) else batch
                x = x.to(device, non_blocking=True)
                x_hat = model(x)
                loss = mse(x_hat, x)
                val_losses.append(float(loss.item()))

        t_loss = float(np.mean(train_losses))
        v_loss = float(np.mean(val_losses))
        scheduler.step(v_loss)
        current_lr = float(optimizer.param_groups[0]["lr"])

        history["train_loss"].append(t_loss)
        history["val_loss"].append(v_loss)
        history["lr"].append(current_lr)

        if cb:
            cb(epoch, t_loss, v_loss)
        else:
            print(f"Epoch {epoch:3d} | train={t_loss:.6f} | val={v_loss:.6f} | lr={current_lr:.2e}")

        if v_loss < best_val_loss:
            best_val_loss = v_loss
            wait = 0
            if save_path:
                save_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), save_path)
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    if save_path and save_path.exists():
        model.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))

    return model, history
