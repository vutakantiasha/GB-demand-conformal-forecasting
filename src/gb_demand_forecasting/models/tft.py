from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from gb_demand_forecasting.utils.io import ensure_dir, write_json

LOGGER = logging.getLogger(__name__)


@dataclass
class Standardizer:
    means: dict[str, float]
    stds: dict[str, float]
    target_mean: float
    target_std: float

    @classmethod
    def fit(cls, df: pd.DataFrame, features: list[str]) -> "Standardizer":
        means: dict[str, float] = {}
        stds: dict[str, float] = {}
        for col in features:
            x = pd.to_numeric(df[col], errors="coerce")
            mean = float(x.mean()) if x.notna().any() else 0.0
            std = float(x.std()) if x.notna().sum() > 1 else 1.0
            if not np.isfinite(std) or std < 1e-8:
                std = 1.0
            means[col] = mean
            stds[col] = std
        y = pd.to_numeric(df["national_demand_mw"], errors="coerce")
        target_mean = float(y.mean())
        target_std = float(y.std())
        if target_std < 1e-8:
            target_std = 1.0
        return cls(means, stds, target_mean, target_std)

    def transform_features(self, frame: pd.DataFrame, features: list[str]) -> np.ndarray:
        cols = []
        for col in features:
            x = pd.to_numeric(frame[col], errors="coerce").fillna(self.means[col])
            cols.append(((x.to_numpy(dtype=np.float32) - self.means[col]) / self.stds[col]))
        return np.stack(cols, axis=1).astype(np.float32)

    def transform_target(self, values: np.ndarray) -> np.ndarray:
        return ((values.astype(np.float32) - self.target_mean) / self.target_std).astype(np.float32)

    def inverse_target(self, values: np.ndarray) -> np.ndarray:
        return values * self.target_std + self.target_mean


class DailyWindowDataset(Dataset):
    """One sample per target day at a fixed previous-day 12:15 forecast origin.

    Encoder: seven complete days of history ending after SP24 on D-1.
    Decoder: SP25-48 on D-1 (24 gap periods) followed by all 48 periods of target day D.
    The 72-step decoder therefore preserves the actual operational forecast origin instead of
    pretending that target-day midnight observations were known in advance.
    """

    def __init__(
        self,
        frame: pd.DataFrame,
        target_dates: list,
        features: list[str],
        scaler: Standardizer,
        encoder_days: int,
        origin_period: int,
        prediction_periods: int = 48,
    ) -> None:
        self.frame = frame.copy()
        self.frame["settlement_date"] = pd.to_datetime(self.frame["settlement_date"]).dt.date
        self.by_date = {d: g.sort_values("settlement_period").copy() for d, g in self.frame.groupby("settlement_date")}
        self.features = features
        self.scaler = scaler
        self.encoder_days = encoder_days
        self.origin_period = origin_period
        self.prediction_periods = prediction_periods
        self.decoder_gap = 48 - origin_period
        self.encoder_len = encoder_days * 48
        self.decoder_len = self.decoder_gap + prediction_periods
        self.samples = [d for d in target_dates if self._valid_date(d)]

    def _valid_date(self, d) -> bool:
        required = [d - timedelta(days=k) for k in range(0, self.encoder_days + 2)]
        if any(day not in self.by_date for day in required):
            return False
        if any(len(self.by_date[day]) != 48 for day in required):
            return False
        return True

    def __len__(self) -> int:
        return len(self.samples)

    def _window(self, d) -> tuple[pd.DataFrame, pd.DataFrame]:
        pieces: list[pd.DataFrame] = []
        start_day = d - timedelta(days=self.encoder_days + 1)
        pieces.append(self.by_date[start_day].query("settlement_period > @self.origin_period"))
        for offset in range(self.encoder_days, 1, -1):
            pieces.append(self.by_date[d - timedelta(days=offset)])
        pieces.append(self.by_date[d - timedelta(days=1)].query("settlement_period <= @self.origin_period"))
        encoder = pd.concat(pieces, ignore_index=True)
        if len(encoder) != self.encoder_len:
            raise RuntimeError(f"Encoder window length {len(encoder)} != {self.encoder_len} for {d}")
        decoder = pd.concat(
            [
                self.by_date[d - timedelta(days=1)].query("settlement_period > @self.origin_period"),
                self.by_date[d],
            ],
            ignore_index=True,
        )
        if len(decoder) != self.decoder_len:
            raise RuntimeError(f"Decoder window length {len(decoder)} != {self.decoder_len} for {d}")
        return encoder, decoder

    def __getitem__(self, idx: int) -> dict[str, Any]:
        d = self.samples[idx]
        encoder, decoder = self._window(d)
        enc_known = self.scaler.transform_features(encoder, self.features)
        enc_y = self.scaler.transform_target(encoder["national_demand_mw"].to_numpy(dtype=np.float32))
        enc_x = np.concatenate([enc_y[:, None], enc_known], axis=1)
        dec_x = self.scaler.transform_features(decoder, self.features)
        y = self.scaler.transform_target(decoder["national_demand_mw"].to_numpy(dtype=np.float32))
        return {
            "encoder": torch.from_numpy(enc_x),
            "decoder": torch.from_numpy(dec_x),
            "target": torch.from_numpy(y),
            "sample_index": torch.tensor(idx, dtype=torch.long),
        }


class GatedLinearUnit(nn.Module):
    def __init__(self, input_size: int, output_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(input_size, output_size * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value, gate = self.proj(x).chunk(2, dim=-1)
        return value * torch.sigmoid(gate)


class GatedResidualNetwork(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, dropout: float) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)
        self.glu = GatedLinearUnit(output_size, output_size)
        self.skip = nn.Linear(input_size, output_size) if input_size != output_size else nn.Identity()
        self.norm = nn.LayerNorm(output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        y = self.fc2(torch.nn.functional.elu(self.fc1(x)))
        y = self.dropout(y)
        y = self.glu(y)
        return self.norm(residual + y)


class VariableSelectionNetwork(nn.Module):
    def __init__(self, n_vars: int, hidden_size: int, dropout: float) -> None:
        super().__init__()
        self.n_vars = n_vars
        self.var_proj = nn.ModuleList(
            [GatedResidualNetwork(1, hidden_size, hidden_size, dropout) for _ in range(n_vars)]
        )
        self.weight_grn = GatedResidualNetwork(n_vars, hidden_size, n_vars, dropout)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        transformed = torch.stack(
            [self.var_proj[i](x[..., i : i + 1]) for i in range(self.n_vars)], dim=-2
        )
        weights = torch.softmax(self.weight_grn(x), dim=-1)
        selected = (transformed * weights.unsqueeze(-1)).sum(dim=-2)
        return selected, weights


class ResearchTemporalFusionTransformer(nn.Module):
    """Self-contained TFT implementation for fixed-origin daily multi-horizon quantile forecasting.

    It implements the principal TFT mechanisms used by this research project: variable selection,
    gated residual networks, recurrent encoder/decoder processing, gated skip connections,
    interpretable multi-head temporal attention, and direct quantile outputs.
    """

    def __init__(
        self,
        n_encoder_vars: int,
        n_decoder_vars: int,
        hidden_size: int,
        heads: int,
        dropout: float,
        quantiles: list[float],
        lstm_layers: int = 1,
    ) -> None:
        super().__init__()
        if hidden_size % heads != 0:
            raise ValueError("hidden_size must be divisible by attention heads")
        self.quantiles = quantiles
        self.enc_vsn = VariableSelectionNetwork(n_encoder_vars, hidden_size, dropout)
        self.dec_vsn = VariableSelectionNetwork(n_decoder_vars, hidden_size, dropout)
        self.encoder_lstm = nn.LSTM(
            hidden_size, hidden_size, num_layers=lstm_layers, batch_first=True, dropout=0.0
        )
        self.decoder_lstm = nn.LSTM(
            hidden_size, hidden_size, num_layers=lstm_layers, batch_first=True, dropout=0.0
        )
        self.enc_gate = GatedResidualNetwork(hidden_size, hidden_size, hidden_size, dropout)
        self.dec_gate = GatedResidualNetwork(hidden_size, hidden_size, hidden_size, dropout)
        self.attention = nn.MultiheadAttention(
            hidden_size, heads, dropout=dropout, batch_first=True
        )
        self.attn_gate = GatedResidualNetwork(hidden_size, hidden_size, hidden_size, dropout)
        self.position_grn = GatedResidualNetwork(hidden_size, hidden_size, hidden_size, dropout)
        self.output = nn.Linear(hidden_size, len(quantiles))

    @staticmethod
    def _mask(dec_len: int, enc_len: int, device: torch.device) -> torch.Tensor:
        mask = torch.zeros((dec_len, enc_len + dec_len), dtype=torch.bool, device=device)
        future = torch.triu(torch.ones((dec_len, dec_len), dtype=torch.bool, device=device), diagonal=1)
        mask[:, enc_len:] = future
        return mask

    def forward(self, encoder: torch.Tensor, decoder: torch.Tensor) -> torch.Tensor:
        enc_selected, _ = self.enc_vsn(encoder)
        dec_selected, _ = self.dec_vsn(decoder)
        enc_lstm, state = self.encoder_lstm(enc_selected)
        dec_lstm, _ = self.decoder_lstm(dec_selected, state)
        enc = self.enc_gate(enc_lstm + enc_selected)
        dec = self.dec_gate(dec_lstm + dec_selected)
        context = torch.cat([enc, dec], dim=1)
        mask = self._mask(dec.size(1), enc.size(1), decoder.device)
        attn, _ = self.attention(dec, context, context, attn_mask=mask, need_weights=False)
        x = self.attn_gate(attn + dec)
        x = self.position_grn(x)
        return self.output(x)


def quantile_loss(pred: torch.Tensor, target: torch.Tensor, quantiles: list[float]) -> torch.Tensor:
    losses = []
    for i, q in enumerate(quantiles):
        error = target - pred[..., i]
        losses.append(torch.maximum((q - 1.0) * error, q * error).unsqueeze(-1))
    return torch.cat(losses, dim=-1).mean()


def _date_list(df: pd.DataFrame, split: str) -> list:
    return sorted(pd.to_datetime(df.loc[df["split"] == split, "settlement_date"]).dt.date.unique())


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def train_tft(
    frame: pd.DataFrame,
    features: list[str],
    cfg: dict[str, Any],
    artifacts_dir: Path,
    seed: int,
    device_name: str = "auto",
    num_workers: int = 0,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    train_rows = frame[frame["split"] == "training"].copy()
    scaler = Standardizer.fit(train_rows, features)
    common = dict(
        frame=frame,
        features=features,
        scaler=scaler,
        encoder_days=int(cfg["encoder_days"]),
        origin_period=int(cfg["origin_period"]),
        prediction_periods=int(cfg.get("prediction_periods", 48)),
    )
    train_ds = DailyWindowDataset(target_dates=_date_list(frame, "training"), **common)
    valid_ds = DailyWindowDataset(target_dates=_date_list(frame, "validation"), **common)
    calib_ds = DailyWindowDataset(target_dates=_date_list(frame, "calibration"), **common)
    test_ds = DailyWindowDataset(target_dates=_date_list(frame, "test"), **common)
    min_samples = int(cfg.get("min_train_samples", 100))
    if len(train_ds) < min_samples:
        raise RuntimeError(f"TFT requires >= {min_samples} train day-windows; found {len(train_ds)}")
    LOGGER.info(
        "TFT windows train=%d val=%d calibration=%d test=%d",
        len(train_ds), len(valid_ds), len(calib_ds), len(test_ds),
    )
    device = _device(device_name)
    model = ResearchTemporalFusionTransformer(
        n_encoder_vars=1 + len(features),
        n_decoder_vars=len(features),
        hidden_size=int(cfg["hidden_size"]),
        heads=int(cfg["attention_heads"]),
        dropout=float(cfg["dropout"]),
        quantiles=[float(q) for q in cfg["quantiles"]],
        lstm_layers=int(cfg.get("lstm_layers", 1)),
    ).to(device)
    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg["batch_size"]),
        shuffle=True,
        num_workers=num_workers,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=int(cfg["batch_size"]),
        shuffle=False,
        num_workers=num_workers,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["learning_rate"]),
        weight_decay=float(cfg.get("weight_decay", 0.0)),
    )
    best_state: dict[str, torch.Tensor] | None = None
    best_val = float("inf")
    patience_left = int(cfg["patience"])
    history: list[dict[str, float]] = []
    quantiles = [float(q) for q in cfg["quantiles"]]
    for epoch in range(1, int(cfg["max_epochs"]) + 1):
        model.train()
        train_losses: list[float] = []
        for batch in train_loader:
            enc = batch["encoder"].to(device)
            dec = batch["decoder"].to(device)
            target = batch["target"].to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(enc, dec)
            loss = quantile_loss(pred, target, quantiles)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["gradient_clip_val"]))
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))
        model.eval()
        valid_losses: list[float] = []
        with torch.no_grad():
            for batch in valid_loader:
                pred = model(batch["encoder"].to(device), batch["decoder"].to(device))
                loss = quantile_loss(pred, batch["target"].to(device), quantiles)
                valid_losses.append(float(loss.cpu()))
        train_loss = float(np.mean(train_losses))
        valid_loss = float(np.mean(valid_losses)) if valid_losses else train_loss
        history.append({"epoch": epoch, "train_quantile_loss": train_loss, "val_quantile_loss": valid_loss})
        LOGGER.info("TFT epoch=%d train=%.5f val=%.5f", epoch, train_loss, valid_loss)
        if valid_loss < best_val - 1e-5:
            best_val = valid_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = int(cfg["patience"])
        else:
            patience_left -= 1
            if patience_left <= 0:
                break
    if best_state is None:
        raise RuntimeError("TFT training did not produce a valid checkpoint")
    model.load_state_dict(best_state)
    model.to(device).eval()

    model_dir = ensure_dir(artifacts_dir / "models" / "tft")
    torch.save(best_state, model_dir / "tft_state_dict.pt")
    pd.DataFrame(history).to_csv(model_dir / "training_history.csv", index=False)
    write_json(asdict(scaler), model_dir / "scaler.json")
    write_json(
        {
            "features": features,
            "config": cfg,
            "best_validation_quantile_loss": best_val,
            "implementation": "self-contained TFT: VSN + GRN + LSTM encoder/decoder + temporal MHA + quantile head",
        },
        model_dir / "model_metadata.json",
    )

    def predict_dataset(ds: DailyWindowDataset, split: str) -> pd.DataFrame:
        loader = DataLoader(ds, batch_size=int(cfg["batch_size"]), shuffle=False, num_workers=num_workers)
        rows: list[dict[str, Any]] = []
        cursor = 0
        with torch.no_grad():
            for batch in loader:
                out = model(batch["encoder"].to(device), batch["decoder"].to(device)).cpu().numpy()
                out = scaler.inverse_target(out)
                batch_size = out.shape[0]
                for b in range(batch_size):
                    target_date = ds.samples[cursor + b]
                    final = out[b, -48:, :]
                    target_frame = ds.by_date[target_date]
                    actual = target_frame["national_demand_mw"].to_numpy(dtype=float)
                    for sp in range(1, 49):
                        rows.append(
                            {
                                "settlement_date": target_date,
                                "settlement_period": sp,
                                "split": split,
                                "national_demand_mw": actual[sp - 1],
                                "model": "tft",
                                "q05": float(final[sp - 1, 0]),
                                "q50": float(final[sp - 1, 1]),
                                "q95": float(final[sp - 1, 2]),
                            }
                        )
                cursor += batch_size
        return pd.DataFrame(rows)

    pred = pd.concat(
        [predict_dataset(calib_ds, "calibration"), predict_dataset(test_ds, "test")],
        ignore_index=True,
    )
    return pred, {"best_validation_quantile_loss": best_val, "train_windows": len(train_ds)}
