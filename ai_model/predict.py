"""
ai_model/predict.py - Inference engine and BaseStrategy integration.

Two entry points are provided:

1. :class:`TransformerPredictor`
   Standalone class that loads a trained checkpoint and returns BUY / HOLD / SELL
   signals (and softmax probabilities) for a raw sequence tensor.

2. :class:`TransformerStrategy`
   A :class:`~trading_agent.BaseStrategy` subclass that plugs the Transformer
   directly into the existing ``TradingAgent`` pipeline without any changes to
   the rest of the codebase.

Usage (standalone)::

    predictor = TransformerPredictor.from_checkpoint("ai_model/checkpoints/best_model.pt")
    signal, probs = predictor.predict_sequence(sequence_tensor)

Usage (strategy integration)::

    from ai_model.predict import TransformerStrategy
    from trading_agent import TradingAgent
    from portfolio import Portfolio

    strategy = TransformerStrategy()
    agent = TradingAgent(portfolio=Portfolio(), strategy=strategy)
    signals_df = agent.run(processed_df)
"""

import logging
import os
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from ai_model.config import ModelConfig
from ai_model.features import build_feature_pipeline, fit_scalers, apply_scalers, clean_features

logger = logging.getLogger(__name__)

# Signal literal type matches trading_agent.Signal
Signal = str   # "BUY" | "SELL" | "HOLD"
_LABEL_TO_SIGNAL = {0: "BUY", 1: "HOLD", 2: "SELL"}


# ---------------------------------------------------------------------------
# Standalone predictor
# ---------------------------------------------------------------------------

class TransformerPredictor:
    """Inference wrapper for a trained :class:`~ai_model.model.TransformerSignalModel`.

    Args:
        model:  Loaded :class:`~ai_model.model.TransformerSignalModel` in eval mode.
        cfg:    :class:`~ai_model.config.ModelConfig` used during training.
        device: Computation device (auto-detected when ``None``).
    """

    def __init__(
        self,
        model: nn.Module,
        cfg: ModelConfig,
        device: Optional[torch.device] = None,
    ) -> None:
        self.model = model
        self.cfg = cfg
        if device is None:
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")
        self.device = device
        self.model.to(self.device)
        self.model.eval()

    # ------------------------------------------------------------------
    # Factory – load from checkpoint
    # ------------------------------------------------------------------

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        device: Optional[torch.device] = None,
    ) -> "TransformerPredictor":
        """Load a predictor from a saved checkpoint.

        Args:
            checkpoint_path: Path to the ``.pt`` checkpoint produced by
                             :func:`~ai_model.train.train`.
            device:          Override computation device.

        Returns:
            Initialised :class:`TransformerPredictor` ready for inference.

        Raises:
            FileNotFoundError: If *checkpoint_path* does not exist.
        """
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_path}. "
                "Train the model first with: python -m ai_model.train"
            )

        # Lazy import to avoid mandatory torch dependency at module import time
        from ai_model.model import TransformerSignalModel

        map_device = device or torch.device("cpu")
        checkpoint = torch.load(checkpoint_path, map_location=map_device, weights_only=False)
        cfg: ModelConfig = checkpoint["cfg"]

        model = TransformerSignalModel(cfg)
        model.load_state_dict(checkpoint["model_state_dict"])
        logger.info(
            "Loaded checkpoint from epoch %d (val_loss=%.4f).",
            checkpoint.get("epoch", -1),
            checkpoint.get("val_loss", float("nan")),
        )
        return cls(model, cfg, device)

    # ------------------------------------------------------------------
    # Prediction helpers
    # ------------------------------------------------------------------

    def predict_sequence(
        self,
        sequence: torch.Tensor,
    ) -> Tuple[Signal, np.ndarray]:
        """Predict the signal for a single pre-built sequence.

        Args:
            sequence: FloatTensor of shape ``(seq_len, num_features)`` **or**
                      ``(1, seq_len, num_features)`` (batch dim optional).

        Returns:
            Tuple of ``(signal_string, probabilities_array)`` where
            *probabilities_array* has shape ``(3,)`` ordered
            [P(BUY), P(HOLD), P(SELL)].
        """
        if sequence.dim() == 2:
            sequence = sequence.unsqueeze(0)    # add batch dim

        sequence = sequence.to(self.device)
        with torch.no_grad():
            logits = self.model(sequence)       # (1, num_classes)
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        pred_idx = int(probs.argmax())
        signal = _LABEL_TO_SIGNAL[pred_idx]
        return signal, probs

    def predict_from_df(
        self,
        df: pd.DataFrame,
        ticker: str,
        scaler_stats: Optional[dict] = None,
    ) -> Tuple[Signal, np.ndarray]:
        """Predict the signal for the *most recent* candles in a DataFrame.

        The function takes the last ``cfg.seq_len`` rows for *ticker* from
        *df* (already normalised if *scaler_stats* is ``None``), builds the
        sequence tensor, and returns the prediction.

        Args:
            df:           Processed DataFrame (output of ``add_indicators()``).
                          Must contain all feature columns.
            ticker:       Ticker symbol to predict for.
            scaler_stats: Pre-fitted scaler stats from training.  If provided,
                          the feature columns are normalised before inference.

        Returns:
            Tuple of ``(signal, probabilities)``.
        """
        ticker_df = df[df["Ticker"] == ticker].copy()
        if ticker_df.empty:
            logger.warning("No rows found for ticker %s – returning HOLD.", ticker)
            return "HOLD", np.array([0.0, 1.0, 0.0])

        # Optionally normalise
        if scaler_stats is not None:
            ticker_df = apply_scalers(ticker_df, self.cfg, scaler_stats)

        ticker_df = ticker_df.sort_values("Datetime").tail(self.cfg.seq_len)
        if len(ticker_df) < self.cfg.seq_len:
            logger.warning(
                "Ticker %s has only %d rows; need %d – returning HOLD.",
                ticker, len(ticker_df), self.cfg.seq_len,
            )
            return "HOLD", np.array([0.0, 1.0, 0.0])

        feature_matrix = ticker_df[self.cfg.feature_columns].to_numpy(dtype=np.float32)
        sequence = torch.tensor(feature_matrix, dtype=torch.float32)
        return self.predict_sequence(sequence)


# ---------------------------------------------------------------------------
# BaseStrategy subclass for TradingAgent integration
# ---------------------------------------------------------------------------

try:
    from trading_agent import BaseStrategy as _BaseStrategy
    _BASE_AVAILABLE = True
except ImportError:
    # Allow predict.py to be imported even outside the project root
    _BASE_AVAILABLE = False
    _BaseStrategy = object  # type: ignore[assignment,misc]


class TransformerStrategy(_BaseStrategy):  # type: ignore[misc]
    """Plug the trained Transformer model into the existing TradingAgent pipeline.

    Drop-in replacement for the rule-based strategies:

    .. code-block:: python

        from ai_model.predict import TransformerStrategy
        from trading_agent import TradingAgent
        from portfolio import Portfolio

        strategy = TransformerStrategy()           # auto-loads best checkpoint
        agent = TradingAgent(Portfolio(), strategy)
        signals_df = agent.run(processed_df)

    Args:
        checkpoint_path: Path to the trained checkpoint.  Defaults to
                         ``ModelConfig().best_model_path``.
        scaler_stats:    Pre-fitted scaler statistics.  ``None`` assumes the
                         DataFrame columns are already normalised.
        confidence_threshold: Minimum probability for the top class to emit a
                              non-HOLD signal.  Reduces over-trading on
                              uncertain predictions.
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        scaler_stats: Optional[dict] = None,
        confidence_threshold: float = 0.5,
    ) -> None:
        if not _BASE_AVAILABLE:
            raise ImportError(
                "trading_agent.BaseStrategy is not available. "
                "Ensure TransformerStrategy is used within the PSE Trading Bot project."
            )

        if checkpoint_path is None:
            checkpoint_path = ModelConfig().best_model_path

        self.predictor = TransformerPredictor.from_checkpoint(checkpoint_path)
        self.scaler_stats = scaler_stats
        self.confidence_threshold = confidence_threshold
        self._df_cache: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # BaseStrategy protocol
    # ------------------------------------------------------------------

    @property
    def lag_columns(self) -> list:
        """No lag columns needed – the Transformer handles its own context."""
        return []

    def set_dataframe(self, df: pd.DataFrame) -> None:
        """Cache the full processed DataFrame for sequence building.

        Call this once before ``TradingAgent.run(df)`` to give the strategy
        access to the full history needed to assemble look-back windows.

        Args:
            df: Processed DataFrame (output of ``add_indicators()``).
        """
        self._df_cache = df.copy()

    def generate_signal(self, row: pd.Series, ticker: str) -> Signal:
        """Generate a BUY / HOLD / SELL signal using the Transformer.

        The strategy looks up all rows for *ticker* up to and including the
        current candle's timestamp from the cached DataFrame, builds the
        ``seq_len`` look-back window, and runs inference.

        Args:
            row:    Current candle row (must contain ``Datetime`` and ``Ticker``).
            ticker: Ticker symbol.

        Returns:
            ``"BUY"``, ``"SELL"``, or ``"HOLD"``.
        """
        if self._df_cache is None:
            logger.warning(
                "TransformerStrategy: no DataFrame cached. "
                "Call strategy.set_dataframe(df) before agent.run(df)."
            )
            return "HOLD"

        current_ts = row.get("Datetime")
        ticker_history = self._df_cache[
            (self._df_cache["Ticker"] == ticker)
            & (self._df_cache["Datetime"] <= current_ts)
        ]

        if ticker_history.empty:
            return "HOLD"

        signal, probs = self.predictor.predict_from_df(
            ticker_history, ticker, self.scaler_stats
        )

        # Apply confidence threshold
        max_prob = float(probs.max())
        if max_prob < self.confidence_threshold:
            return "HOLD"

        return signal
