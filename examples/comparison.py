from __future__ import annotations

from statistics import mean
from time import perf_counter

from heston_pricer import EuropeanOption, HestonNeuralNetworkPricer, HestonParameters, HestonPricer


def build_test_cases() -> list[tuple[EuropeanOption, HestonParameters]]:
    return [
        (
            EuropeanOption(spot=100.0, strike=100.0, maturity=1.0, rate=0.03, dividend_yield=0.0, is_call=True),
            HestonParameters(v0=0.04, theta=0.04, kappa=2.0, sigma=0.5, rho=-0.7),
        ),
        (
            EuropeanOption(spot=95.0, strike=100.0, maturity=0.5, rate=0.025, dividend_yield=0.01, is_call=True),
            HestonParameters(v0=0.05, theta=0.06, kappa=1.7, sigma=0.45, rho=-0.5),
        ),
        (
            EuropeanOption(spot=120.0, strike=110.0, maturity=1.5, rate=0.035, dividend_yield=0.0, is_call=True),
            HestonParameters(v0=0.03, theta=0.05, kappa=2.4, sigma=0.65, rho=-0.6),
        ),
        (
            EuropeanOption(spot=80.0, strike=85.0, maturity=2.0, rate=0.04, dividend_yield=0.015, is_call=False),
            HestonParameters(v0=0.06, theta=0.05, kappa=1.3, sigma=0.55, rho=-0.75),
        ),
    ]


def benchmark(pricer, test_cases, repeats: int = 25):
    prices = []
    timings = []

    for option, params in test_cases:
        start = perf_counter()
        result = None
        for _ in range(repeats):
            result = pricer.price(option, params)
        elapsed = perf_counter() - start
        prices.append(result)
        timings.append(elapsed / repeats)

    return prices, timings


def main() -> None:
    exact_pricer = HestonPricer()
    neural_pricer = HestonNeuralNetworkPricer()

    print("Training neural surrogate from the exact engine...")
    neural_pricer.fit(n_samples=1000, seed=7)

    test_cases = build_test_cases()

    exact_prices, exact_timings = benchmark(exact_pricer, test_cases)
    neural_prices, neural_timings = benchmark(neural_pricer, test_cases)

    absolute_errors = [abs(exact - neural) for exact, neural in zip(exact_prices, neural_prices)]
    relative_errors = [abs(exact - neural) / max(abs(exact), 1e-8) for exact, neural in zip(exact_prices, neural_prices)]

    primary_option, primary_params = test_cases[0]
    neural_greeks = neural_pricer.greeks(primary_option, primary_params)
    exact_greeks = exact_pricer.greeks(primary_option, primary_params)

    print("\nComparison summary")
    print("=" * 80)
    print(f"Test cases: {len(test_cases)}")
    print(f"Average exact pricing time:  {mean(exact_timings) * 1_000:.4f} ms")
    print(f"Average neural pricing time: {mean(neural_timings) * 1_000:.4f} ms")
    print(f"Speedup: {mean(exact_timings) / max(mean(neural_timings), 1e-12):.2f}x")
    print(f"Mean absolute error:  {mean(absolute_errors):.6f}")
    print(f"Mean relative error:   {mean(relative_errors):.6%}")

    print("\nPrimary option")
    print(primary_option)
    print(primary_params)
    print(f"Exact price:  {exact_prices[0]:.6f}")
    print(f"Neural price: {neural_prices[0]:.6f}")
    print(f"Absolute error: {absolute_errors[0]:.6f}")

    print("\nExact Greeks")
    print(exact_greeks)
    print("Neural Greeks")
    print(neural_greeks)

    print("\nPer-case comparison")
    for index, ((option, params), exact_price, neural_price, exact_time, neural_time, abs_error, rel_error) in enumerate(
        zip(test_cases, exact_prices, neural_prices, exact_timings, neural_timings, absolute_errors, relative_errors),
        start=1,
    ):
        print(f"Case {index}:")
        print(f"  option = {option}")
        print(f"  params = {params}")
        print(f"  exact_price = {exact_price:.6f}")
        print(f"  neural_price = {neural_price:.6f}")
        print(f"  abs_error = {abs_error:.6f}")
        print(f"  rel_error = {rel_error:.6%}")
        print(f"  exact_time = {exact_time * 1_000:.4f} ms")
        print(f"  neural_time = {neural_time * 1_000:.4f} ms")


if __name__ == "__main__":
    main()