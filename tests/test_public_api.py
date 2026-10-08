from heston_pricer import EuropeanOption, HestonParameters, HestonPricer
from heston_pricer import core


def test_public_types_construct():
    option = EuropeanOption(spot=100.0, strike=105.0, maturity=1.0, rate=0.03)
    params = HestonParameters(v0=0.04, theta=0.04, kappa=2.0, sigma=0.5, rho=-0.7)

    assert option.spot == 100.0
    assert params.kappa == 2.0


def test_native_backend_is_available():
    assert core.backend_name() == "cpp_heston_analytic"


def test_native_backend_supports_batch_pricing():
    option = EuropeanOption(spot=100.0, strike=100.0, maturity=1.0, rate=0.03)
    params = HestonParameters(v0=0.04, theta=0.04, kappa=2.0, sigma=0.5, rho=-0.7)

    prices = core.price_european_options([(option, params), (option, params)])
    greeks = core.greeks_european_options([(option, params)])

    assert len(prices) == 2
    assert prices[0] == prices[1]
    assert len(greeks) == 1
    assert greeks[0]["price"] == prices[0]


def test_pricer_prices_european_option():
    pricer = HestonPricer()
    option = EuropeanOption(spot=100.0, strike=100.0, maturity=1.0, rate=0.03)
    params = HestonParameters(v0=0.04, theta=0.04, kappa=2.0, sigma=0.5, rho=-0.7)

    price = pricer.price(option, params)
    greeks = pricer.greeks(option, params)

    assert price > 0.0
    assert greeks.price == price
    assert greeks.delta != 0.0
    assert greeks.gamma > 0.0
