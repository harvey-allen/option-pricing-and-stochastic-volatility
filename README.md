# Heston Pricer

Heston Pricer is a research project exploring fast, accurate European option pricing under a Heston-style stochastic volatility model.

The original goal was to answer a practical risk-engine question: can a learned surrogate reproduce the behavior of a much slower pricing model while remaining accurate enough for use in a trading or risk workflow?

This project compares two approaches:

- an exact native pricing engine used as the reference implementation
- a neural-network surrogate trained from the exact engine to approximate prices and Greeks more quickly

The focus is on European options and on building a workflow that can support further work on calibration, latency reduction, and model validation.

## Background

The starting point for this work was my master's dissertation research found in the legacy folder. That exploratory version was useful for testing ideas, but it was not structured as a reusable pricing system.

This repository turns that research into a cleaner library-style codebase so the exact model and the neural surrogate can be benchmarked side by side. The comparison is intended to show the trade-off between speed and accuracy, and to highlight where a surrogate model may be useful in a low-latency production risk setting.
