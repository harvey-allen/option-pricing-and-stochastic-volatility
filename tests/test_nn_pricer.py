from heston_pricer import EuropeanOption, HestonNeuralNetworkPricer, HestonParameters


def test_neural_pricer_trains_and_prices():
    pricer = HestonNeuralNetworkPricer()
    pricer.fit(n_samples=60, seed=11)

    option = EuropeanOption(spot=100.0, strike=100.0, maturity=1.0, rate=0.03)
    params = HestonParameters(v0=0.04, theta=0.04, kappa=2.0, sigma=0.5, rho=-0.7)

    price = pricer.price(option, params)
    greeks = pricer.greeks(option, params)

    assert isinstance(price, float)
    assert greeks.price == price
    assert greeks.gamma != 0.0
