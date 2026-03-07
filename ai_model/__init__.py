"""
ai_model - Transformer-based trade signal prediction for the PSE Trading Bot.

This package provides all modules required to train, evaluate, and deploy a
lightweight Transformer model that predicts BUY / HOLD / SELL signals from
1-minute OHLCV + indicator sequences.

Modules
-------
config      Hyperparameters and path settings for the AI model.
features    Feature engineering pipeline (OHLCV + indicators → normalised sequences).
dataset     PyTorch Dataset class with windowed sequences and look-ahead labels.
model       Lightweight Transformer encoder architecture.
train       Training loop with AdamW optimiser, LR scheduler, and checkpointing.
evaluate    Evaluation metrics: accuracy, classification report, Sharpe, drawdown.
predict     Inference engine and ``BaseStrategy`` subclass for live integration.

Quick-start
-----------
>>> from ai_model.model import TransformerSignalModel
>>> from ai_model.config import ModelConfig
>>> cfg = ModelConfig()
>>> net = TransformerSignalModel(cfg)
"""

from ai_model.config import ModelConfig

# TransformerSignalModel requires PyTorch; import it lazily so that the
# config and feature-engineering modules remain usable in environments where
# torch is not yet installed.
try:
    from ai_model.model import TransformerSignalModel
    __all__ = ["ModelConfig", "TransformerSignalModel"]
except ImportError:
    __all__ = ["ModelConfig"]
