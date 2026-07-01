"""Fast one-class MobileNetV2 anomaly detector (no autoencoder)."""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v2

from src.config import BASE_CHANNELS, LATENT_DIM, LATENT_L1


def _init_weights(module):
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(module.weight, nonlinearity="leaky_relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class ConvAutoencoder(nn.Module):
    """
    Backward-compatible class name.
    Actually a one-class anomaly scorer trained with pseudo anomalies.
    """

    def __init__(self, latent_dim: int = LATENT_DIM, base_channels: int = BASE_CHANNELS):
        super().__init__()
        c = max(16, int(base_channels))
        self.latent_dim = int(latent_dim)
        self.base_channels = c

        width_mult = max(0.35, min(1.0, c / 32.0))
        backbone = mobilenet_v2(weights=None, width_mult=width_mult)
        first_out = int(backbone.features[0][0].out_channels)
        first_conv = nn.Conv2d(1, first_out, kernel_size=3, stride=2, padding=1, bias=False)
        with torch.no_grad():
            first_conv.weight.copy_(backbone.features[0][0].weight.mean(dim=1, keepdim=True))
        backbone.features[0][0] = first_conv
        self.encoder = backbone.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        last_ch = self.encoder[-1][0].out_channels
        self.proj = nn.Linear(last_ch, self.latent_dim, bias=False)
        self.norm = nn.LayerNorm(self.latent_dim)
        self.head = nn.Linear(self.latent_dim, 1)

        self.apply(_init_weights)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        h = self.pool(h).flatten(1)
        z = self.proj(h)
        z = self.norm(z)
        return z

    def forward(self, x: torch.Tensor):
        z = self.encode(x)
        logits = self.head(z).squeeze(1)
        return logits, z

    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """
        Kept for compatibility with existing evaluation/check code.
        Returns anomaly score in [0, 1], higher = more anomalous.
        """
        logits, _ = self(x)
        return torch.sigmoid(logits)


def _make_pseudo_anomaly(x: torch.Tensor) -> torch.Tensor:
    """Strong corruption to create pseudo-anomalies from normal spectrograms."""
    out = x.clone()
    n, _, f, t = out.shape
    dev = x.device

    # Random frequency mask (fully vectorized)
    mask_f = torch.randint(low=max(2, f // 16), high=max(3, f // 4), size=(n, 1, 1, 1), device=dev)
    max_start_f = (f - mask_f.float()).clamp(min=1)
    start_f = (torch.rand(n, 1, 1, 1, device=dev) * max_start_f).long()
    ar_f = torch.arange(f, device=dev)[None, None, :, None]
    fmask = (ar_f >= start_f) & (ar_f < start_f + mask_f)
    out = torch.where(fmask, torch.tensor(0.0, device=dev), out)

    # Random time mask (fully vectorized)
    mask_t = torch.randint(low=max(2, t // 16), high=max(3, t // 4), size=(n, 1, 1, 1), device=dev)
    max_start_t = (t - mask_t.float()).clamp(min=1)
    start_t = (torch.rand(n, 1, 1, 1, device=dev) * max_start_t).long()
    ar_t = torch.arange(t, device=dev)[None, None, None, :]
    tmask = (ar_t >= start_t) & (ar_t < start_t + mask_t)
    out = torch.where(tmask, torch.tensor(0.0, device=dev), out)

    # Additive noise + random gain
    noise = 0.1 * torch.randn_like(out)
    gain = torch.empty((n, 1, 1, 1), device=dev).uniform_(0.7, 1.4)
    out = torch.clamp(out * gain + noise, 0.0, 1.0)
    return out


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
    latent_l1: float = LATENT_L1,
    stop_event=None,
):
    """Train one-class scorer via normal-vs-pseudo-anomaly objective."""
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=max(3, min(10, patience // 2))
    )
    bce = nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    max_grad_norm = 5.0

    best_val_loss = float("inf")
    wait = 0
    history = {"train_loss": [], "val_loss": [], "lr": []}

    def _step_loss(x_batch: torch.Tensor):
        pos_logits, pos_z = model(x_batch)
        x_neg = _make_pseudo_anomaly(x_batch)
        neg_logits, neg_z = model(x_neg)

        y_pos = torch.zeros_like(pos_logits)
        y_neg = torch.ones_like(neg_logits)
        cls_loss = bce(pos_logits, y_pos) + bce(neg_logits, y_neg)

        # Encourage separation between normal and pseudo-anomaly scores.
        margin = F.relu(0.3 + torch.sigmoid(pos_logits) - torch.sigmoid(neg_logits)).mean()
        reg = latent_l1 * (pos_z.abs().mean() + neg_z.abs().mean())
        return cls_loss + 0.5 * margin + reg

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
                    loss = _step_loss(x)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss = _step_loss(x)
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
                        v_loss = _step_loss(x)
                else:
                    v_loss = _step_loss(x)
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
            print(f"Epoch {epoch:3d} | train={t_loss:.4f} | val={v_loss:.4f} | lr={current_lr:.2e}")

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
