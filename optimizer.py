"""
optimizer.py - Strategy parameter optimiser for the PSE Trading Bot.

Uses an **adaptive random hill-climbing** algorithm to search for strategy
and risk parameters that maximise total return on a given historical dataset.

Algorithm overview
------------------
1. Start with the default (or user-supplied) parameter values.
2. Evaluate the objective (total return %) by running a full backtest.
3. For each of *n_iterations* iterations:
   a. With probability *exploration_prob*, sample a completely random
      parameter set within the defined bounds (exploration).
   b. Otherwise, randomly perturb the current-best parameters by a
      Gaussian offset that shrinks linearly over the run (exploitation).
   c. Run a backtest with the candidate parameters.
   d. If the candidate return is better, promote it to the current best.
4. Return the best parameters, metrics, and a full iteration history.

This approach is conceptually similar to gradient descent (iterative,
improving objective) but is applicable to the non-differentiable, discrete
backtesting simulation.  Parameter bounds keep every explored configuration
within human-interpretable ranges.
"""

import logging
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import pandas as pd

import config
import indicators as ind
from backtester import Backtester
from trading_agent import BollingerBandStrategy, EMACrossoverStrategy, RSIStrategy

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Parameter bound descriptor
# ---------------------------------------------------------------------------

@dataclass
class ParameterBounds:
    """Declares the valid range and default value for a single optimisable
    parameter.

    Args:
        name: Parameter name (matches the key used in the params dict).
        min_val: Minimum allowed value (inclusive).
        max_val: Maximum allowed value (inclusive).
        initial: Starting / default value.
        is_int: If True the value is rounded to the nearest integer.
    """

    name: str
    min_val: float
    max_val: float
    initial: float
    is_int: bool = False

    def clip(self, val: float) -> float:
        """Clip *val* to [min_val, max_val] and optionally round to int."""
        clipped = float(np.clip(val, self.min_val, self.max_val))
        return float(round(clipped)) if self.is_int else clipped

    def perturb(self, val: float, scale: float, rng: np.random.Generator) -> float:
        """Return a randomly perturbed copy of *val* still within bounds.

        The perturbation magnitude is *scale* × (max_val − min_val) drawn
        from a standard normal, so *scale* = 0.1 means perturbations are
        roughly ±10 % of the parameter's full range.
        """
        range_size = self.max_val - self.min_val
        delta = rng.normal(0.0, scale * range_size)
        if self.is_int:
            delta = float(round(delta))
        return self.clip(val + delta)

    def random_sample(self, rng: np.random.Generator) -> float:
        """Uniformly sample a random value within bounds."""
        if self.is_int:
            return float(int(rng.integers(int(self.min_val), int(self.max_val) + 1)))
        return float(rng.uniform(self.min_val, self.max_val))


# ---------------------------------------------------------------------------
# Per-strategy parameter bounds
# ---------------------------------------------------------------------------

#: Human-reasonable bounds for each strategy.
#: Extending this dict is the only change needed to add a new strategy.
STRATEGY_PARAM_BOUNDS: dict[str, list[ParameterBounds]] = {
    "EMA Crossover": [
        ParameterBounds("ema_fast", 2, 20, config.EMA_FAST, is_int=True),
        ParameterBounds("ema_slow", 10, 100, config.EMA_SLOW, is_int=True),
        ParameterBounds("max_position_pct", 0.01, 0.20, config.MAX_POSITION_PCT),
        ParameterBounds("stop_loss_pct", 0.005, 0.10, config.STOP_LOSS_PCT),
        ParameterBounds("take_profit_pct", 0.01, 0.20, config.TAKE_PROFIT_PCT),
    ],
    "RSI Mean-Reversion": [
        ParameterBounds("rsi_period", 5, 30, config.RSI_PERIOD, is_int=True),
        ParameterBounds("rsi_oversold", 10.0, 45.0, config.RSI_OVERSOLD),
        ParameterBounds("rsi_overbought", 55.0, 90.0, config.RSI_OVERBOUGHT),
        ParameterBounds("max_position_pct", 0.01, 0.20, config.MAX_POSITION_PCT),
        ParameterBounds("stop_loss_pct", 0.005, 0.10, config.STOP_LOSS_PCT),
        ParameterBounds("take_profit_pct", 0.01, 0.20, config.TAKE_PROFIT_PCT),
    ],
    "Bollinger Bands": [
        ParameterBounds("bollinger_period", 5, 50, config.BOLLINGER_PERIOD, is_int=True),
        ParameterBounds("bollinger_std", 1.0, 3.5, config.BOLLINGER_STD),
        ParameterBounds("max_position_pct", 0.01, 0.20, config.MAX_POSITION_PCT),
        ParameterBounds("stop_loss_pct", 0.005, 0.10, config.STOP_LOSS_PCT),
        ParameterBounds("take_profit_pct", 0.01, 0.20, config.TAKE_PROFIT_PCT),
    ],
}


# ---------------------------------------------------------------------------
# Optimisation result container
# ---------------------------------------------------------------------------

@dataclass
class OptimizationResult:
    """Container for all outputs of a completed optimisation run.

    Attributes:
        strategy_name: Name of the strategy that was optimised.
        best_params: Parameter dict that yielded the highest return.
        best_return_pct: Total return achieved with *best_params*.
        initial_params: The parameter values used as the starting point.
        initial_return_pct: Total return achieved with *initial_params*.
        iteration_history: List of ``(iteration, best_return_pct, best_params)``
            tuples recorded at the end of every iteration.
        n_iterations: Number of optimisation iterations executed.
        n_evaluations: Total number of backtest evaluations performed.
    """

    strategy_name: str
    best_params: dict
    best_return_pct: float
    initial_params: dict
    initial_return_pct: float
    iteration_history: list
    n_iterations: int
    n_evaluations: int


# ---------------------------------------------------------------------------
# Optimiser
# ---------------------------------------------------------------------------

class StrategyOptimizer:
    """Optimise strategy + risk parameters to maximise total return.

    Each candidate parameter set is evaluated by computing technical
    indicators with the candidate values and running a full backtest on
    *df_raw*.

    Args:
        df_raw: Raw OHLCV DataFrame (not yet indicator-enriched).  This is
            the dataset the optimiser uses as its ground truth — typically
            the same data loaded for the active date range.
        strategy_name: One of the keys in :data:`STRATEGY_PARAM_BOUNDS`.
        initial_capital: Starting portfolio capital in PHP.
        n_iterations: Maximum number of optimisation iterations.
        exploration_prob: Probability (0–1) of exploring a fully random
            parameter set instead of perturbing the current best.
            Higher values encourage broader exploration; lower values
            focus exploitation around the current best.
        perturbation_scale: Controls the magnitude of parameter
            perturbations.  Expressed as a fraction of each parameter's
            total range; decays linearly from this value × 2 to this
            value × 0.5 over the run.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        df_raw: pd.DataFrame,
        strategy_name: str,
        initial_capital: float = config.INITIAL_CAPITAL,
        n_iterations: int = 50,
        exploration_prob: float = 0.2,
        perturbation_scale: float = 0.1,
        seed: int = 42,
    ) -> None:
        if strategy_name not in STRATEGY_PARAM_BOUNDS:
            raise ValueError(
                f"Unknown strategy '{strategy_name}'. "
                f"Valid options: {list(STRATEGY_PARAM_BOUNDS)}"
            )

        self.df_raw = df_raw
        self.strategy_name = strategy_name
        self.initial_capital = initial_capital
        self.n_iterations = n_iterations
        self.exploration_prob = exploration_prob
        self.perturbation_scale = perturbation_scale
        self.rng = np.random.default_rng(seed)
        self.param_bounds: list[ParameterBounds] = STRATEGY_PARAM_BOUNDS[strategy_name]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        progress_callback: Optional[Callable[[int, int, float, dict], None]] = None,
    ) -> OptimizationResult:
        """Run the optimisation and return the best parameters found.

        Args:
            progress_callback: Optional callable invoked after each
                iteration with ``(iteration, total_iterations,
                best_return_pct, best_params)``.  Useful for updating a
                progress bar in the UI.

        Returns:
            :class:`OptimizationResult` with the best parameters and full
            iteration history.
        """
        initial_params = self._initial_params()
        initial_return = self._evaluate(initial_params)

        current_params = initial_params.copy()
        current_return = initial_return

        best_params = initial_params.copy()
        best_return = initial_return

        history: list[tuple] = [(0, best_return, best_params.copy())]
        n_eval = 1

        logger.info(
            "Optimisation started: strategy=%s  initial_return=%.2f%%  n_iter=%d",
            self.strategy_name, initial_return, self.n_iterations,
        )

        for i in range(1, self.n_iterations + 1):
            # Perturbation scale decays linearly so early iterations
            # explore broadly and later iterations refine the best region.
            scale = self.perturbation_scale * (2.0 - 1.5 * i / self.n_iterations)

            # Exploration: jump to a completely random point
            if self.rng.random() < self.exploration_prob:
                candidate = self._random_params()
            else:
                candidate = self._perturb_params(current_params, scale)

            candidate_return = self._evaluate(candidate)
            n_eval += 1

            # Hill-climbing: only accept improvements
            if candidate_return > current_return:
                current_params = candidate
                current_return = candidate_return

            if current_return > best_return:
                best_params = current_params.copy()
                best_return = current_return

            history.append((i, best_return, best_params.copy()))

            logger.debug(
                "Iter %d/%d: candidate=%.2f%%  best=%.2f%%",
                i, self.n_iterations, candidate_return, best_return,
            )

            if progress_callback is not None:
                progress_callback(i, self.n_iterations, best_return, best_params)

        logger.info(
            "Optimisation complete: best_return=%.2f%%  evaluations=%d",
            best_return, n_eval,
        )

        return OptimizationResult(
            strategy_name=self.strategy_name,
            best_params=best_params,
            best_return_pct=best_return,
            initial_params=initial_params,
            initial_return_pct=initial_return,
            iteration_history=history,
            n_iterations=self.n_iterations,
            n_evaluations=n_eval,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _evaluate(self, params: dict) -> float:
        """Run a backtest with *params* and return the total return (%).

        Returns -9999 on any exception so a bad parameter set never crashes
        the optimisation loop.
        """
        try:
            df_ind = ind.add_indicators_custom(
                self.df_raw,
                ema_fast=int(params.get("ema_fast", config.EMA_FAST)),
                ema_slow=int(params.get("ema_slow", config.EMA_SLOW)),
                rsi_period=int(params.get("rsi_period", config.RSI_PERIOD)),
                bollinger_period=int(params.get("bollinger_period", config.BOLLINGER_PERIOD)),
                bollinger_std=float(params.get("bollinger_std", config.BOLLINGER_STD)),
            )

            strategy = self._build_strategy(params)

            bt = Backtester(
                initial_capital=self.initial_capital,
                strategy=strategy,
                max_position_pct=float(params.get("max_position_pct", config.MAX_POSITION_PCT)),
                stop_loss_pct=float(params.get("stop_loss_pct", config.STOP_LOSS_PCT)),
                take_profit_pct=float(params.get("take_profit_pct", config.TAKE_PROFIT_PCT)),
            )
            metrics = bt.run(df_ind)
            return float(metrics["total_return_pct"])
        except Exception as exc:  # pragma: no cover
            logger.warning("Evaluation failed for params %s: %s", params, exc)
            return -9999.0

    def _build_strategy(self, params: dict):
        """Construct the appropriate strategy instance from *params*."""
        if self.strategy_name == "EMA Crossover":
            return EMACrossoverStrategy(
                fast_period=int(params.get("ema_fast", config.EMA_FAST)),
                slow_period=int(params.get("ema_slow", config.EMA_SLOW)),
            )
        if self.strategy_name == "RSI Mean-Reversion":
            return RSIStrategy(
                oversold=float(params.get("rsi_oversold", config.RSI_OVERSOLD)),
                overbought=float(params.get("rsi_overbought", config.RSI_OVERBOUGHT)),
            )
        # Bollinger Bands strategy reads the BB columns directly; the period
        # and std are baked into the indicator columns at evaluation time.
        return BollingerBandStrategy()

    def _initial_params(self) -> dict:
        """Return the default parameter values defined by the bound descriptors."""
        return {b.name: b.initial for b in self.param_bounds}

    def _perturb_params(self, params: dict, scale: float) -> dict:
        """Return a copy of *params* with each value randomly perturbed."""
        return {
            b.name: b.perturb(params[b.name], scale, self.rng)
            for b in self.param_bounds
        }

    def _random_params(self) -> dict:
        """Return a uniformly random parameter set within all bounds."""
        return {b.name: b.random_sample(self.rng) for b in self.param_bounds}
