from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Iterable

import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from . import core
except ImportError:  # pragma: no cover - native extension not built yet
    core = None
from .models import EuropeanOption, GreekResults, HestonParameters


@dataclass(frozen=True)
class NeuralPricerConfig:
    hidden_layer_sizes: tuple[int, ...] = (128, 128, 64)
    activation: str = "relu"
    alpha: float = 1e-4
    learning_rate_init: float = 1e-3
    max_iter: int = 500
    random_state: int = 42
    solver: str = "lbfgs"


def _option_to_features(option: EuropeanOption, params: HestonParameters) -> np.ndarray:
    spot = max(option.spot, 1e-8)
    strike = max(option.strike, 1e-8)
    maturity = max(option.maturity, 1e-8)
    return np.array(
        [
            np.log(spot / strike),
            np.log(spot),
            np.log(strike),
            np.sqrt(maturity),
            maturity,
            option.rate,
            option.dividend_yield,
            1.0 if option.is_call else 0.0,
            params.v0,
            params.theta,
            params.kappa,
            params.sigma,
            params.rho,
        ],
        dtype=float,
    )


def _normal_cdf(value: float) -> float:
    return 0.5 * math.erfc(-value / math.sqrt(2.0))


def _black_scholes_price(option: EuropeanOption, volatility: float) -> float:
    spot = max(option.spot, 1e-8)
    strike = max(option.strike, 1e-8)
    maturity = max(option.maturity, 1e-8)
    volatility = max(volatility, 1e-8)

    sigma_root_t = volatility * np.sqrt(maturity)
    if sigma_root_t < 1e-12:
        discounted_spot = spot * np.exp(-option.dividend_yield * maturity)
        discounted_strike = strike * np.exp(-option.rate * maturity)
        intrinsic = max(discounted_spot - discounted_strike, 0.0)
        return intrinsic if option.is_call else intrinsic - discounted_spot + discounted_strike

    drift = option.rate - option.dividend_yield
    d1 = (np.log(spot / strike) + (drift + 0.5 * volatility * volatility) * maturity) / sigma_root_t
    d2 = d1 - sigma_root_t
    discounted_spot = spot * np.exp(-option.dividend_yield * maturity)
    discounted_strike = strike * np.exp(-option.rate * maturity)
    call = discounted_spot * _normal_cdf(d1) - discounted_strike * _normal_cdf(d2)
    return call if option.is_call else call - discounted_spot + discounted_strike


def _residual_target(option: EuropeanOption, params: HestonParameters) -> float:
    if core is None:
        raise NotImplementedError("Native Heston pricing core has not been built yet.")
    exact_price = core.price_european_option(option, params)
    baseline = _black_scholes_price(option, np.sqrt(max(params.v0, 1e-8)))
    return exact_price - baseline


class HestonNeuralNetworkPricer:
    def __init__(self, config: NeuralPricerConfig | None = None) -> None:
        self.config = config or NeuralPricerConfig()
        self._model: Pipeline | None = None

    @property
    def fitted(self) -> bool:
        return self._model is not None

    def fit(self, training_examples: Iterable[tuple[EuropeanOption, HestonParameters]] | None = None, *, n_samples: int = 5000, seed: int | None = None) -> "HestonNeuralNetworkPricer":
        if training_examples is None:
            training_examples, targets = self._generate_training_data(n_samples=n_samples, seed=seed)
        else:
            examples = list(training_examples)
            if not examples:
                raise ValueError("training_examples must not be empty.")
            targets = np.array([_residual_target(option, params) for option, params in examples], dtype=float)
            training_examples = examples

        features = np.vstack([_option_to_features(option, params) for option, params in training_examples])

        estimator = MLPRegressor(
            hidden_layer_sizes=self.config.hidden_layer_sizes,
            activation=self.config.activation,
            alpha=self.config.alpha,
            learning_rate_init=self.config.learning_rate_init,
            max_iter=self.config.max_iter,
            random_state=self.config.random_state if seed is None else seed,
            shuffle=True,
            solver=self.config.solver,
            early_stopping=self.config.solver != "lbfgs",
            validation_fraction=0.15 if self.config.solver != "lbfgs" else 0.1,
            n_iter_no_change=25,
        )

        self._model = TransformedTargetRegressor(
            regressor=Pipeline([
            ("scaler", StandardScaler()),
            ("regressor", estimator),
        ]),
            transformer=StandardScaler(),
        )
        self._model.fit(features, targets)
        return self

    def price(self, option: EuropeanOption, params: HestonParameters) -> float:
        if self._model is None:
            raise NotImplementedError("Train or load the neural pricer before calling price().")
        features = _option_to_features(option, params).reshape(1, -1)
        baseline = _black_scholes_price(option, np.sqrt(max(params.v0, 1e-8)))
        return float(baseline + self._model.predict(features)[0])

    def greeks(self, option: EuropeanOption, params: HestonParameters, *, bumps: dict[str, float] | None = None) -> GreekResults:
        if self._model is None:
            raise NotImplementedError("Train or load the neural pricer before calling greeks().")

        bumps = bumps or {}
        spot_bump = bumps.get("spot", max(1e-4, abs(option.spot) * 1e-4))
        variance_bump = bumps.get("v0", max(1e-5, abs(params.v0) * 1e-4))
        maturity_bump = bumps.get("maturity", max(1e-5, abs(option.maturity) * 1e-4))
        rho_bump = bumps.get("rho", 1e-4)

        base = self.price(option, params)

        spot_up = self.price(
            EuropeanOption(option.spot + spot_bump, option.strike, option.maturity, option.rate, option.dividend_yield, option.is_call),
            params,
        )
        spot_down = self.price(
            EuropeanOption(max(option.spot - spot_bump, 1e-8), option.strike, option.maturity, option.rate, option.dividend_yield, option.is_call),
            params,
        )

        variance_up = self.price(
            option,
            HestonParameters(params.v0 + variance_bump, params.theta, params.kappa, params.sigma, params.rho),
        )
        variance_down = self.price(
            option,
            HestonParameters(max(params.v0 - variance_bump, 1e-8), params.theta, params.kappa, params.sigma, params.rho),
        )

        maturity_up = self.price(
            EuropeanOption(option.spot, option.strike, option.maturity + maturity_bump, option.rate, option.dividend_yield, option.is_call),
            params,
        )
        maturity_down = self.price(
            EuropeanOption(option.spot, option.strike, max(option.maturity - maturity_bump, 1e-8), option.rate, option.dividend_yield, option.is_call),
            params,
        )

        rho_up = self.price(
            option,
            HestonParameters(params.v0, params.theta, params.kappa, params.sigma, min(params.rho + rho_bump, 0.999999)),
        )
        rho_down = self.price(
            option,
            HestonParameters(params.v0, params.theta, params.kappa, params.sigma, max(params.rho - rho_bump, -0.999999)),
        )

        return GreekResults(
            price=base,
            delta=(spot_up - spot_down) / (2.0 * spot_bump),
            gamma=(spot_up - 2.0 * base + spot_down) / (spot_bump * spot_bump),
            vega=(variance_up - variance_down) / (2.0 * variance_bump),
            theta=(maturity_down - maturity_up) / (2.0 * maturity_bump),
            rho=(rho_up - rho_down) / (2.0 * rho_bump),
        )

    def save(self, path: str | Path) -> None:
        if self._model is None:
            raise NotImplementedError("Train or load the neural pricer before calling save().")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        import joblib

        joblib.dump({"config": self.config, "model": self._model}, path)

    @classmethod
    def load(cls, path: str | Path) -> "HestonNeuralNetworkPricer":
        import joblib

        payload = joblib.load(Path(path))
        pricer = cls(payload["config"])
        pricer._model = payload["model"]
        return pricer

    def _generate_training_data(self, *, n_samples: int, seed: int | None) -> tuple[list[tuple[EuropeanOption, HestonParameters]], np.ndarray]:
        rng = np.random.default_rng(self.config.random_state if seed is None else seed)
        examples: list[tuple[EuropeanOption, HestonParameters]] = []
        targets: list[float] = []

        for _ in range(n_samples):
            option = EuropeanOption(
                spot=float(rng.uniform(50.0, 150.0)),
                strike=float(rng.uniform(50.0, 150.0)),
                maturity=float(rng.uniform(0.05, 2.5)),
                rate=float(rng.uniform(0.0, 0.10)),
                dividend_yield=float(rng.uniform(0.0, 0.05)),
                is_call=bool(rng.integers(0, 2)),
            )
            params = HestonParameters(
                v0=float(rng.uniform(0.01, 0.25)),
                theta=float(rng.uniform(0.01, 0.25)),
                kappa=float(rng.uniform(0.5, 4.0)),
                sigma=float(rng.uniform(0.1, 1.2)),
                rho=float(rng.uniform(-0.95, 0.0)),
            )
            examples.append((option, params))
            targets.append(_residual_target(option, params))

        return examples, np.asarray(targets, dtype=float)