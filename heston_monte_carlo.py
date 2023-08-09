import numpy as np


def heston_model_monte_carlo_simulation(
    s0, v0, rho, kappa, theta, sigma, t, n, m, r=0.02
):
    """
    Inputs:
     - S0, v0: Initial parameters for asset and variance.
     - rho: Correlation between asset returns and variance.
     - kappa: Rate of mean reversion in variance process.
     - theta: Long-term mean of variance process.
     - sigma: Vol of vol / volatility of variance process.
     - T: Time of simulation.
     - N: Number of time steps.
     - M: Number of scenarios / simulations.
     - R: Risk Free Rate
    Outputs:
     - S: Asset prices over time (array).
     - V: Variance over time (array).
    """

    dt = t / n
    mu = np.array([0, 0])
    cov = np.array([[1, rho], [rho, 1]])
    S = np.full(shape=(n + 1, m), fill_value=s0)
    V = np.full(shape=(n + 1, m), fill_value=v0)
    Z = np.random.multivariate_normal(mu, cov, (n, m))

    for i in range(1, n + 1):
        S[i] = S[i - 1] * np.exp(
            (r - 0.5 * V[i - 1]) * dt + np.sqrt(V[i - 1] * dt) * Z[i - 1, :, 0]
        )
        V[i] = np.maximum(
            V[i - 1]
            + kappa * (theta - V[i - 1]) * dt
            + sigma * np.sqrt(V[i - 1] * dt) * Z[i - 1, :, 1],
            0,
        )

    return S, V
