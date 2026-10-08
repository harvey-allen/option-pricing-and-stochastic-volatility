#include <pybind11/pybind11.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

namespace {

constexpr double kPi = 3.141592653589793238462643383279502884;
constexpr double kSmallNumber = 1e-12;

struct EuropeanOptionData {
    double spot;
    double strike;
    double maturity;
    double rate;
    double dividend_yield;
    bool is_call;
};

struct HestonParametersData {
    double v0;
    double theta;
    double kappa;
    double sigma;
    double rho;
};

struct GreekResultData {
    double price;
    double delta;
    double gamma;
    double vega;
    double theta;
    double rho;
};

template <typename T>
T extract_field(const py::handle &object, const char *name) {
    if (py::hasattr(object, name)) {
        return py::getattr(object, name).cast<T>();
    }

    if (py::isinstance<py::dict>(object)) {
        py::dict mapping = py::reinterpret_borrow<py::dict>(object);
        py::handle key = py::str(name);
        if (mapping.contains(key)) {
            return mapping[key].cast<T>();
        }
    }

    throw py::value_error(std::string("Missing required field: ") + name);
}

EuropeanOptionData extract_option(const py::handle &object) {
    EuropeanOptionData option{};
    option.spot = extract_field<double>(object, "spot");
    option.strike = extract_field<double>(object, "strike");
    option.maturity = extract_field<double>(object, "maturity");
    option.rate = extract_field<double>(object, "rate");
    option.dividend_yield = extract_field<double>(object, "dividend_yield");
    option.is_call = extract_field<bool>(object, "is_call");
    return option;
}

HestonParametersData extract_parameters(const py::handle &object) {
    HestonParametersData parameters{};
    parameters.v0 = extract_field<double>(object, "v0");
    parameters.theta = extract_field<double>(object, "theta");
    parameters.kappa = extract_field<double>(object, "kappa");
    parameters.sigma = extract_field<double>(object, "sigma");
    parameters.rho = extract_field<double>(object, "rho");
    return parameters;
}

double simpson_integral(const std::function<double(double)> &function, double lower, double upper, int intervals) {
    if (intervals < 2) {
        intervals = 2;
    }
    if (intervals % 2 != 0) {
        ++intervals;
    }

    const double step = (upper - lower) / static_cast<double>(intervals);
    double sum = function(lower) + function(upper);

    for (int index = 1; index < intervals; ++index) {
        const double u = lower + step * static_cast<double>(index);
        sum += function(u) * (index % 2 == 0 ? 2.0 : 4.0);
    }

    return sum * step / 3.0;
}

std::complex<double> heston_characteristic_function(
    const EuropeanOptionData &option,
    const HestonParametersData &parameters,
    std::complex<double> u
) {
    using complex = std::complex<double>;
    const complex i(0.0, 1.0);

    const double variance = std::max(parameters.v0, kSmallNumber);
    const double theta = std::max(parameters.theta, kSmallNumber);
    const double sigma = std::max(parameters.sigma, kSmallNumber);
    const double maturity = std::max(option.maturity, kSmallNumber);
    const double x = std::log(std::max(option.spot, kSmallNumber));
    const double drift = option.rate - option.dividend_yield;

    const complex alpha = -(u * u) - i * u;
    const complex beta = parameters.kappa - parameters.rho * sigma * i * u;
    const complex discriminant = std::sqrt(beta * beta - sigma * sigma * alpha);
    const complex g = (beta - discriminant) / (beta + discriminant);
    const complex exp_term = std::exp(-discriminant * maturity);

    const complex c_term = i * u * drift * maturity +
        (parameters.kappa * theta / (sigma * sigma)) *
            ((beta - discriminant) * maturity - 2.0 * std::log((1.0 - g * exp_term) / (1.0 - g)));

    const complex d_term = (beta - discriminant) / (sigma * sigma) * ((1.0 - exp_term) / (1.0 - g * exp_term));

    return std::exp(c_term + d_term * variance + i * u * x);
}

double probability_p2(const EuropeanOptionData &option, const HestonParametersData &parameters) {
    const double strike = std::max(option.strike, kSmallNumber);
    const std::complex<double> i(0.0, 1.0);
    const double maturity = std::max(option.maturity, kSmallNumber);
    const double upper = 120.0 + 20.0 * std::sqrt(maturity);
    const int intervals = 1000;
    const double lower = 1e-8;

    const auto integrand = [&](double u_value) {
        const std::complex<double> u(u_value, 0.0);
        const std::complex<double> characteristic = heston_characteristic_function(option, parameters, u);
        const std::complex<double> kernel = std::exp(-i * u * std::log(strike)) * characteristic / (i * u);
        return std::real(kernel);
    };

    const double integral = simpson_integral(integrand, lower, upper, intervals);
    return 0.5 + integral / kPi;
}

double probability_p1(const EuropeanOptionData &option, const HestonParametersData &parameters) {
    const double strike = std::max(option.strike, kSmallNumber);
    const std::complex<double> i(0.0, 1.0);
    const double maturity = std::max(option.maturity, kSmallNumber);
    const double upper = 120.0 + 20.0 * std::sqrt(maturity);
    const int intervals = 1000;
    const double lower = 1e-8;

    const std::complex<double> shifted_denominator = heston_characteristic_function(
        option,
        parameters,
        std::complex<double>(0.0, -1.0)
    );

    if (std::abs(shifted_denominator) < kSmallNumber) {
        throw std::runtime_error("Heston characteristic function became unstable while computing P1.");
    }

    const auto integrand = [&](double u_value) {
        const std::complex<double> u(u_value, 0.0);
        const std::complex<double> shifted = heston_characteristic_function(option, parameters, u - i) / shifted_denominator;
        const std::complex<double> kernel = std::exp(-i * u * std::log(strike)) * shifted / (i * u);
        return std::real(kernel);
    };

    const double integral = simpson_integral(integrand, lower, upper, intervals);
    return 0.5 + integral / kPi;
}

double black_scholes_price(const EuropeanOptionData &option, double volatility) {
    const double spot = std::max(option.spot, kSmallNumber);
    const double strike = std::max(option.strike, kSmallNumber);
    const double maturity = std::max(option.maturity, kSmallNumber);
    const double drift = option.rate - option.dividend_yield;
    const double sigma_root_t = volatility * std::sqrt(maturity);

    if (sigma_root_t < kSmallNumber) {
        const double discounted_spot = spot * std::exp(-option.dividend_yield * maturity);
        const double discounted_strike = strike * std::exp(-option.rate * maturity);
        const double intrinsic = option.is_call ? std::max(discounted_spot - discounted_strike, 0.0)
                                                 : std::max(discounted_strike - discounted_spot, 0.0);
        return intrinsic;
    }

    const double d1 = (std::log(spot / strike) + (drift + 0.5 * volatility * volatility) * maturity) / sigma_root_t;
    const double d2 = d1 - sigma_root_t;

    const auto norm_cdf = [](double value) {
        return 0.5 * std::erfc(-value / std::sqrt(2.0));
    };

    const double discounted_spot = spot * std::exp(-option.dividend_yield * maturity);
    const double discounted_strike = strike * std::exp(-option.rate * maturity);
    const double call = discounted_spot * norm_cdf(d1) - discounted_strike * norm_cdf(d2);

    if (option.is_call) {
        return call;
    }

    return call - discounted_spot + discounted_strike;
}

double price_european_option_impl(const EuropeanOptionData &option, const HestonParametersData &parameters) {
    if (option.maturity <= 0.0) {
        const double discounted_spot = option.spot;
        const double intrinsic = option.is_call ? std::max(discounted_spot - option.strike, 0.0)
                                                 : std::max(option.strike - discounted_spot, 0.0);
        return intrinsic;
    }

    if (option.spot <= 0.0 || option.strike <= 0.0) {
        throw std::invalid_argument("Spot and strike must be positive.");
    }

    if (parameters.sigma <= kSmallNumber) {
        return black_scholes_price(option, std::sqrt(std::max(parameters.v0, kSmallNumber)));
    }

    const double p1 = probability_p1(option, parameters);
    const double p2 = probability_p2(option, parameters);

    const double discounted_spot = option.spot * std::exp(-option.dividend_yield * option.maturity);
    const double discounted_strike = option.strike * std::exp(-option.rate * option.maturity);
    const double call_price = discounted_spot * p1 - discounted_strike * p2;

    if (option.is_call) {
        return call_price;
    }

    return call_price - discounted_spot + discounted_strike;
}

double bump_value(double value) {
    const double magnitude = std::max(std::abs(value), 1.0);
    return std::max(1e-5, magnitude * 1e-4);
}

GreekResultData compute_greeks(const EuropeanOptionData &option, const HestonParametersData &parameters) {
    const double base_price = price_european_option_impl(option, parameters);

    const double spot_bump = bump_value(option.spot);
    const double variance_bump = bump_value(parameters.v0);
    const double maturity_bump = std::min(std::max(option.maturity * 1e-4, 1e-5), std::max(option.maturity * 0.5, 1e-5));
    const double rho_bump = std::max(1e-5, (1.0 - std::abs(parameters.rho)) * 1e-4);

    const EuropeanOptionData spot_up_option{option.spot + spot_bump, option.strike, option.maturity, option.rate, option.dividend_yield, option.is_call};
    const EuropeanOptionData spot_down_option{std::max(option.spot - spot_bump, kSmallNumber), option.strike, option.maturity, option.rate, option.dividend_yield, option.is_call};

    const HestonParametersData variance_up_params{parameters.v0 + variance_bump, parameters.theta, parameters.kappa, parameters.sigma, parameters.rho};
    const HestonParametersData variance_down_params{std::max(parameters.v0 - variance_bump, kSmallNumber), parameters.theta, parameters.kappa, parameters.sigma, parameters.rho};

    const EuropeanOptionData maturity_up_option{option.spot, option.strike, option.maturity + maturity_bump, option.rate, option.dividend_yield, option.is_call};
    const EuropeanOptionData maturity_down_option{option.spot, option.strike, std::max(option.maturity - maturity_bump, kSmallNumber), option.rate, option.dividend_yield, option.is_call};

    const HestonParametersData rho_up_params{parameters.v0, parameters.theta, parameters.kappa, parameters.sigma, std::min(parameters.rho + rho_bump, 0.999999)};
    const HestonParametersData rho_down_params{parameters.v0, parameters.theta, parameters.kappa, parameters.sigma, std::max(parameters.rho - rho_bump, -0.999999)};

    const double spot_up_price = price_european_option_impl(spot_up_option, parameters);
    const double spot_down_price = price_european_option_impl(spot_down_option, parameters);
    const double variance_up_price = price_european_option_impl(option, variance_up_params);
    const double variance_down_price = price_european_option_impl(option, variance_down_params);
    const double maturity_up_price = price_european_option_impl(maturity_up_option, parameters);
    const double maturity_down_price = price_european_option_impl(maturity_down_option, parameters);
    const double rho_up_price = price_european_option_impl(option, rho_up_params);
    const double rho_down_price = price_european_option_impl(option, rho_down_params);

    const double delta = (spot_up_price - spot_down_price) / (2.0 * spot_bump);
    const double gamma = (spot_up_price - 2.0 * base_price + spot_down_price) / (spot_bump * spot_bump);
    const double vega = (variance_up_price - variance_down_price) / (2.0 * variance_bump);
    const double theta = (maturity_down_price - maturity_up_price) / (2.0 * maturity_bump);
    const double rho = (rho_up_price - rho_down_price) / (2.0 * rho_bump);

    return GreekResultData{base_price, delta, gamma, vega, theta, rho};
}

std::pair<EuropeanOptionData, HestonParametersData> extract_case(const py::handle &object) {
    py::sequence case_item = py::reinterpret_borrow<py::sequence>(object);
    if (py::len(case_item) != 2) {
        throw py::value_error("Each batch item must be a pair of (option, parameters).");
    }

    return {extract_option(case_item[0]), extract_parameters(case_item[1])};
}

}  // namespace

PYBIND11_MODULE(core, m) {
    m.doc() = "Native Heston pricing core for European options.";

    m.def("price_european_option", [](py::object option, py::object parameters) {
        const auto option_data = extract_option(option);
        const auto parameter_data = extract_parameters(parameters);
        double price = 0.0;
        {
            py::gil_scoped_release release;
            price = price_european_option_impl(option_data, parameter_data);
        }
        return price;
    });

    m.def("greeks_european_option", [](py::object option, py::object parameters) {
        const auto option_data = extract_option(option);
        const auto parameter_data = extract_parameters(parameters);
        GreekResultData greeks{};
        {
            py::gil_scoped_release release;
            greeks = compute_greeks(option_data, parameter_data);
        }

        py::dict result;
        result["price"] = greeks.price;
        result["delta"] = greeks.delta;
        result["gamma"] = greeks.gamma;
        result["vega"] = greeks.vega;
        result["theta"] = greeks.theta;
        result["rho"] = greeks.rho;
        return result;
    });

    m.def("price_european_options", [](py::object cases) {
        std::vector<std::pair<EuropeanOptionData, HestonParametersData>> batch;
        for (py::handle item : py::reinterpret_borrow<py::iterable>(cases)) {
            batch.push_back(extract_case(item));
        }

        std::vector<double> prices;
        prices.reserve(batch.size());
        {
            py::gil_scoped_release release;
            for (const auto &[option_data, parameter_data] : batch) {
                prices.push_back(price_european_option_impl(option_data, parameter_data));
            }
        }

        py::list results;
        for (double price : prices) {
            results.append(price);
        }
        return results;
    });

    m.def("greeks_european_options", [](py::object cases) {
        std::vector<std::pair<EuropeanOptionData, HestonParametersData>> batch;
        for (py::handle item : py::reinterpret_borrow<py::iterable>(cases)) {
            batch.push_back(extract_case(item));
        }

        std::vector<GreekResultData> results_payload;
        results_payload.reserve(batch.size());
        {
            py::gil_scoped_release release;
            for (const auto &[option_data, parameter_data] : batch) {
                results_payload.push_back(compute_greeks(option_data, parameter_data));
            }
        }

        py::list results;
        for (const auto &greeks : results_payload) {
            py::dict result;
            result["price"] = greeks.price;
            result["delta"] = greeks.delta;
            result["gamma"] = greeks.gamma;
            result["vega"] = greeks.vega;
            result["theta"] = greeks.theta;
            result["rho"] = greeks.rho;
            results.append(result);
        }
        return results;
    });

    m.def("backend_name", []() { return "cpp_heston_analytic"; });
}
