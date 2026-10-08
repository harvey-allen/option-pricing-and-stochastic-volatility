from __future__ import annotations

from collections.abc import Mapping

from .models import EuropeanOption, GreekResults, HestonParameters

try:
    from . import _core
except ImportError:  # pragma: no cover - native extension not built yet
    _core = None


class HestonPricer:
    def price(self, option: EuropeanOption, params: HestonParameters) -> float:
        if _core is None:
            raise NotImplementedError("Native Heston pricing core has not been built yet.")
        return _core.price_european_option(option, params)

    def greeks(self, option: EuropeanOption, params: HestonParameters) -> GreekResults:
        if _core is None:
            raise NotImplementedError("Native Heston pricing core has not been built yet.")
        greeks = _core.greeks_european_option(option, params)
        if isinstance(greeks, GreekResults):
            return greeks
        if isinstance(greeks, Mapping):
            return GreekResults(**greeks)
        if isinstance(greeks, (tuple, list)) and len(greeks) == 6:
            return GreekResults(*greeks)
        raise TypeError("Unexpected Greeks payload returned by native Heston core.")
