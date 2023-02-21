using Pkg

Pkg.activate(".")

using MarketClearing

using DataFrames

function calculate_tweaked_scenario_metrics(wind_powers::Vector{Float64}, tweaked_pdf::Vector{Float64})
    mean = sum(wind_powers .* tweaked_pdf)
    variance = sum((wind_powers .- mean) .^ 2 .* tweaked_pdf)
    return mean, variance
end

overrides = [
    Dict("TWEAKED_CDF" => Dict("gamma" => 0.25)),
    Dict("TWEAKED_CDF" => Dict("gamma" => 0.50)),
    Dict("TWEAKED_CDF" => Dict("gamma" => 0.75)),
    Dict("TWEAKED_CDF" => Dict("gamma" => 1.00)),
    Dict("TWEAKED_CDF" => Dict("gamma" => 2.00)),
    Dict("TWEAKED_CDF" => Dict("gamma" => 3.00)),
    Dict("TWEAKED_CDF" => Dict("gamma" => 5.00)),
    Dict("TWEAKED_CDF" => Dict("delta" => 0.25)),
    Dict("TWEAKED_CDF" => Dict("delta" => 0.50)),
    Dict("TWEAKED_CDF" => Dict("delta" => 0.75)),
    Dict("TWEAKED_CDF" => Dict("delta" => 1.00)),
    Dict("TWEAKED_CDF" => Dict("delta" => 2.00)),
    Dict("TWEAKED_CDF" => Dict("delta" => 3.00)),
    Dict("TWEAKED_CDF" => Dict("delta" => 5.00)),
]

results = DataFrames.DataFrame(
    delta=Any[],
    gamma=Any[],
    mean=Any[],
    variance=Any[],
    supply_day_ahead=Any[],
    consumption_day_ahead=Any[],
    social_welfare=Any[],
    price_day_ahead=Any[],
)

for override in overrides
    settings = MarketClearing.get_settings(; overrides=override)
    wind_powers, reference_pdf, tweaked_pdf = MarketClearing.build_data(
        settings.SAMPLES.mean,
        settings.SAMPLES.variance,
        settings.SAMPLES.num_samples,
        settings.GENERAL.seed,
        settings.TWEAKED_CDF.gamma,
        settings.TWEAKED_CDF.delta,
    )
    mean, variance = calculate_tweaked_scenario_metrics(wind_powers, tweaked_pdf)
    utility = MarketClearing.Utility(settings.UTILITY_FUNCTION.tau, settings.UTILITY_FUNCTION.beta)
    loads = [MarketClearing.Load(1000, reference_pdf, utility)]
    cost = MarketClearing.Cost(settings.COST_FUNCTION.alpha)
    generators = [MarketClearing.Generator(1000, tweaked_pdf, cost)]
    social_welfare, supply_day_ahead, consumption_day_ahead, price_day_ahead = MarketClearing.solve_centralized_model(wind_powers, loads, generators)
    push!(results, (settings.TWEAKED_CDF.delta, settings.TWEAKED_CDF.gamma, mean, variance, supply_day_ahead, consumption_day_ahead, social_welfare, price_day_ahead))
end

println(results)