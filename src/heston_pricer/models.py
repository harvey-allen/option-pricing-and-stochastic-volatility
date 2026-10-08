from dataclasses import dataclass


@dataclass(frozen=True)
class EuropeanOption:
    spot: float
    strike: float
    maturity: float
    rate: float
    dividend_yield: float = 0.0
    is_call: bool = True


@dataclass(frozen=True)
class HestonParameters:
    v0: float
    theta: float
    kappa: float
    sigma: float
    rho: float


@dataclass(frozen=True)
class GreekResults:
    price: float
    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float
