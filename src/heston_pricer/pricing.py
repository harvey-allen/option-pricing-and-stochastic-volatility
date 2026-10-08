from __future__ import annotations

from collections.abc import Mapping

from .models import EuropeanOption, GreekResults, HestonParameters

try:
    from . import core
except ImportError:  # pragma: no cover - native extension not built yet
    core = None


class HestonPricer:
    def price(self, option: EuropeanOption, params: HestonParameters) -> float:
        if core is None:
            raise NotImplementedError("Native Heston pricing core has not been built yet.")
        return core.price_european_option(option, params)

    def price_many(self, cases: list[tuple[EuropeanOption, HestonParameters]]) -> list[float]:
        if core is None:
            raise NotImplementedError("Native Heston pricing core has not been built yet.")
        return list(core.price_european_options(cases))

    def greeks(self, option: EuropeanOption, params: HestonParameters) -> GreekResults:
        if core is None:
            raise NotImplementedError("Native Heston pricing core has not been built yet.")
        greeks = core.greeks_european_option(option, params)
        if isinstance(greeks, GreekResults):
            return greeks
        if isinstance(greeks, Mapping):
            return GreekResults(**greeks)
        if isinstance(greeks, (tuple, list)) and len(greeks) == 6:
            return GreekResults(*greeks)
        raise TypeError("Unexpected Greeks payload returned by native Heston core.")

    def greeks_many(self, cases: list[tuple[EuropeanOption, HestonParameters]]) -> list[GreekResults]:
        if core is None:
            raise NotImplementedError("Native Heston pricing core has not been built yet.")
        payload = core.greeks_european_options(cases)
        return [GreekResults(**item) if isinstance(item, dict) else GreekResults(*item) for item in payload]
