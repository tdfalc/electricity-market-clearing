using Pkg

Pkg.activate(".")

using MarketClearing

function print_results(header, social_welfare, supply_day_ahead, consumption_day_ahead, price_day_ahead)
    @info join(
        (
            "$(header):",
            "* Social welfare => $(social_welfare);",
            "* Supply => $(supply_day_ahead);",
            "* Consumption => $(consumption_day_ahead);",
            "* Price => $(price_day_ahead);"
        ),
        "\n"
    )
end

settings = MarketClearing.get_settings()

wind_powers, reference_pdf, tweaked_pdf = MarketClearing.build_data(
    settings.SAMPLES.mean,
    settings.SAMPLES.variance,
    settings.SAMPLES.num_samples,
    settings.GENERAL.seed,
    settings.TWEAKED_CDF.gamma,
    settings.TWEAKED_CDF.delta,
)

loads = [MarketClearing.Load(1000, reference_pdf, MarketClearing.Utility(settings.UTILITY_FUNCTION.tau, settings.UTILITY_FUNCTION.beta))]
generators = [MarketClearing.Generator(1000, tweaked_pdf, MarketClearing.Cost(settings.COST_FUNCTION.alpha))]

# Note: Centralized market clearing - The market operator collects bids of agents and finds 
# Note: socially optimal contracts at the day-ahead stage, followed by real-time recourse decisions.
social_welfare, supply_day_ahead, consumption_day_ahead, price_day_ahead = MarketClearing.solve_centralized_model(
    wind_powers,
    loads,
    generators,
)
print_results("Centralized", social_welfare, supply_day_ahead, consumption_day_ahead, price_day_ahead)

# Note: Competitive equilibrium - Price-setting agent optimizes a set of quilibrium prices
# Note: in response to the value of the system imbalance for each outcome of renewable  production.
social_welfare, supply_day_ahead, consumption_day_ahead, price_day_ahead = MarketClearing.solve_mixed_complementarity_model(
    wind_powers,
    loads,
    generators,
)
print_results("Mixed Complementarity", social_welfare, supply_day_ahead, consumption_day_ahead, price_day_ahead)

social_welfare, supply_day_ahead, consumption_day_ahead, price_day_ahead = MarketClearing.solve_iterative_model(
    wind_powers, loads, generators,
    settings.EQUILIBRIUM.max_iterations,
    settings.EQUILIBRIUM.step_size,
    settings.EQUILIBRIUM.convergence_tolerance,
)
print_results("Iterative", social_welfare, supply_day_ahead, consumption_day_ahead, price_day_ahead)
