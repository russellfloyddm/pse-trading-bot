"""
ai_model/model.py - Lightweight Transformer encoder for trade signal prediction.

Architecture overview
---------------------

    Input  (batch, seq_len, num_features)
      │
      ├─ Linear projection → (batch, seq_len, d_model)
      │
      ├─ Positional Encoding (sinusoidal, added in-place)
      │
      ├─ Transformer Encoder  (N × EncoderLayer)
      │     • Multi-head Self-Attention (nhead heads)
      │     • Position-wise Feed-Forward Network
      │     • LayerNorm + residual connections
      │     • Dropout
      │
      ├─ Mean pooling over the time dimension → (batch, d_model)
      │
      └─ Classification Head (Linear → LayerNorm → GELU → Linear)
                          → (batch, num_classes)

Parameter count (default config: d_model=64, nhead=4, layers=2, ffn=128)
~roughly 130 k parameters — intentionally lightweight for fast iteration.

Classes
-------
PositionalEncoding      Sinusoidal position embeddings (Vaswani et al. 2017).
TransformerSignalModel  Full end-to-end model.
"""

import math

import torch
import torch.nn as nn

from ai_model.config import ModelConfig


# ---------------------------------------------------------------------------
# Positional encoding
# ---------------------------------------------------------------------------

class PositionalEncoding(nn.Module):
    """Add sinusoidal positional encodings to the input embeddings.

    Args:
        d_model: Embedding dimension.
        dropout: Dropout probability applied after adding the encoding.
        max_len: Maximum supported sequence length.
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 512) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Pre-compute the encoding matrix once.
        pe = torch.zeros(max_len, d_model)                    # (max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()  # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)                                  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to *x*.

        Args:
            x: Input tensor of shape ``(batch, seq_len, d_model)``.

        Returns:
            Tensor of the same shape with positional encoding added.
        """
        x = x + self.pe[:, : x.size(1), :]   # type: ignore[index]
        return self.dropout(x)


# ---------------------------------------------------------------------------
# Transformer signal model
# ---------------------------------------------------------------------------

class TransformerSignalModel(nn.Module):
    """Lightweight Transformer encoder for BUY / HOLD / SELL classification.

    Args:
        cfg: :class:`~ai_model.config.ModelConfig` instance that provides all
             architecture hyperparameters.

    Example::

        cfg = ModelConfig()
        model = TransformerSignalModel(cfg)
        x = torch.randn(32, cfg.seq_len, cfg.num_features)   # (batch, seq, feats)
        logits = model(x)                                     # (32, 3)
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg

        # -- Input projection --------------------------------------------------
        # Project raw features into the model's embedding space.
        self.input_proj = nn.Linear(cfg.num_features, cfg.d_model)

        # -- Positional encoding -----------------------------------------------
        self.pos_enc = PositionalEncoding(cfg.d_model, dropout=cfg.dropout)

        # -- Transformer encoder -----------------------------------------------
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,       # expects (batch, seq, d_model)
            norm_first=True,        # pre-norm for more stable training
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=cfg.num_encoder_layers,
            enable_nested_tensor=False,
        )

        # -- Classification head -----------------------------------------------
        # Mean-pool the sequence, then classify with a two-layer MLP.
        self.classifier = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.LayerNorm(cfg.d_model),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model, cfg.num_classes),
        )

        self._init_weights()

    # ------------------------------------------------------------------
    # Weight initialisation
    # ------------------------------------------------------------------

    def _init_weights(self) -> None:
        """Xavier-uniform initialisation for linear layers."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute class logits for a batch of sequences.

        Args:
            x: FloatTensor of shape ``(batch, seq_len, num_features)``.

        Returns:
            FloatTensor of shape ``(batch, num_classes)`` with raw logits.
        """
        # 1. Project features to embedding dimension
        x = self.input_proj(x)                   # (B, T, d_model)

        # 2. Add positional encoding
        x = self.pos_enc(x)                      # (B, T, d_model)

        # 3. Transformer encoder
        x = self.transformer_encoder(x)          # (B, T, d_model)

        # 4. Mean pool across the time dimension
        x = x.mean(dim=1)                        # (B, d_model)

        # 5. Classification head
        logits = self.classifier(x)              # (B, num_classes)
        return logits

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
