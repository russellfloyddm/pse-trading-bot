"""
ai_model/config.py - Hyperparameters and path settings for the Transformer model.

All training, architecture, and data settings live here so that they can be
changed in a single place without touching training or inference code.
"""

import os
from dataclasses import dataclass, field
from typing import List

# ---------------------------------------------------------------------------
# Resolve paths relative to this file so the package works regardless of the
# working directory from which the user launches scripts.
# ---------------------------------------------------------------------------
_AI_MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_AI_MODEL_DIR)


@dataclass
class ModelConfig:
    """All configurable hyperparameters for the Transformer signal model.

    Attributes
    ----------
    Feature & sequence settings
        feature_columns:   Names of input feature columns (must exist in the
                           processed DataFrame after ``add_indicators()``).
        seq_len:           Number of 1-minute candles fed as context (look-back
                           window).
        label_horizon:     Number of candles ahead used to compute the label.
        label_threshold:   Minimum future return (decimal) to qualify as BUY or
                           SELL; otherwise HOLD.

    Architecture hyperparameters
        d_model:           Embedding dimension for the Transformer.
        nhead:             Number of attention heads (must divide d_model evenly).
        num_encoder_layers: Stacked Transformer encoder layers.
        dim_feedforward:   Inner dimension of the position-wise FFN.
        dropout:           Dropout probability applied in encoder and embedding.
        num_classes:       Number of output classes (3: BUY=0, HOLD=1, SELL=2).

    Training settings
        batch_size:        Mini-batch size.
        num_epochs:        Maximum training epochs.
        learning_rate:     Initial learning rate for AdamW.
        weight_decay:      L2 regularisation coefficient.
        patience:          Early-stopping patience in epochs.
        val_split:         Fraction of training data used for validation.
        test_split:        Fraction of all data held out for final evaluation.
        random_seed:       Random seed for reproducibility.

    Paths
        processed_data_file: Path to the processed CSV produced by the main
                             pipeline (``data_pipeline`` + ``indicators``).
        checkpoint_dir:      Directory where model checkpoints are saved.
        best_model_path:     File path for the best checkpoint.
    """

    # ------------------------------------------------------------------
    # Feature / sequence settings
    # ------------------------------------------------------------------
    feature_columns: List[str] = field(default_factory=lambda: [
        "Open", "High", "Low", "Close", "Volume",
        "EMA_9", "EMA_21",
        "RSI",
        "BB_upper", "BB_middle", "BB_lower",
        "Returns", "Volatility",
    ])
    seq_len: int = 30                   # 30-minute look-back window
    label_horizon: int = 5              # predict 5-minute ahead return
    label_threshold: float = 0.002      # ±0.2% return threshold for BUY/SELL

    # ------------------------------------------------------------------
    # Architecture hyperparameters
    # ------------------------------------------------------------------
    d_model: int = 64                   # embedding dimension
    nhead: int = 4                      # attention heads
    num_encoder_layers: int = 2         # stacked encoder layers (lightweight)
    dim_feedforward: int = 128          # FFN inner dimension
    dropout: float = 0.1
    num_classes: int = 3                # BUY=0, HOLD=1, SELL=2

    # ------------------------------------------------------------------
    # Training settings
    # ------------------------------------------------------------------
    batch_size: int = 64
    num_epochs: int = 50
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 10                  # early-stopping patience
    val_split: float = 0.15
    test_split: float = 0.15
    random_seed: int = 42

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------
    processed_data_file: str = os.path.join(
        _PROJECT_ROOT, "data", "processed", "market_data_processed.csv"
    )
    checkpoint_dir: str = os.path.join(_AI_MODEL_DIR, "checkpoints")
    best_model_path: str = os.path.join(_AI_MODEL_DIR, "checkpoints", "best_model.pt")

    def __post_init__(self) -> None:
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    @property
    def num_features(self) -> int:
        """Number of input features per time step."""
        return len(self.feature_columns)
