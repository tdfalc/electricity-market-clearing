function solve_iterative_model(
    wind_powers::Vector{Float64},
    loads::Vector{Load},
    generators::Vector{Generator},
    max_iterations::Int,
    step_size::Float64,
    convergence_tolerance::Float64
)

    market_prices = zeros(max_iterations, length(wind_powers))
    scenarios = 1:length(wind_powers)

    for i in 1:max_iterations-1

        supply_day_ahead_total, supply_real_time, supply_day_ahead_cost, supply_real_time_cost = solve_supplier_problem(wind_powers, market_prices[i, :], generators)

        consumption_day_ahead_total, consumption_real_time, consumption_day_ahead_utility, consumption_real_time_utility = solve_consumption_problem(wind_powers, market_prices[i, :], loads)

        social_welfare = consumption_day_ahead_utility + consumption_real_time_utility - supply_day_ahead_cost - supply_real_time_cost

        power_balance_discrepancies = 0
        for scenario in scenarios
            power_balance = (
                wind_powers[scenario] +
                supply_day_ahead_total +
                sum(supply_real_time[supplier, scenario] for supplier in 1:length(generators)) -
                consumption_day_ahead_total -
                sum(consumption_real_time[consumer, scenario] for consumer in 1:length(loads))
            )

            market_prices[i+1, scenario] = market_prices[i, scenario] - step_size * power_balance
            power_balance_discrepancies += norm(power_balance)
        end

        if norm(power_balance_discrepancies) <= convergence_tolerance
            equilibrium_price = sum(market_prices[i, scenario] for scenario in scenarios)
            return social_welfare, supply_day_ahead_total, consumption_day_ahead_total, equilibrium_price
        end
    end
    return
end

function solve_supplier_problem(wind_powers::Vector{Float64}, market_prices::Vector{Float64}, generators::Vector{MarketClearing.Generator})

    optimizer = optimizer_with_attributes(Ipopt.Optimizer, ("print_level" => 0))
    model = Model(optimizer)

    scenarios = 1:length(wind_powers)
    suppliers = 1:length(generators)

    @variable(model, supply_day_ahead[suppliers])
    @variable(model, supply_real_time[suppliers, scenarios])

    @constraint(model, [supplier in suppliers], 0 <= supply_day_ahead[supplier] <= generators[supplier].capacity)
    @constraint(model, [supplier in suppliers, scenario in scenarios], -generators[supplier].capacity <= supply_real_time[supplier, scenario] <= generators[supplier].capacity)

    @objective(model, Max,
        sum(sum(market_prices[scenario] * (supply_day_ahead[supplier] + supply_real_time[supplier, scenario]) for scenario in scenarios) for supplier in suppliers) -
        sum(generators[supplier].cost.calculate(supply_day_ahead[supplier]) for supplier in suppliers) -
        sum(sum(generators[supplier].probabilities[scenario] * generators[supplier].cost.calculate(supply_real_time[supplier, scenario]) for scenario in scenarios) for supplier in suppliers),
    )

    JuMP.optimize!(model)

    supply_day_ahead_total = sum([JuMP.value(supply_day_ahead[supplier]) for supplier in suppliers])
    supply_real_time = JuMP.value.(supply_real_time)

    supply_day_ahead_cost = sum(generators[supplier].cost.calculate(JuMP.value(supply_day_ahead[supplier])) for supplier in suppliers)
    supply_real_time_cost = sum(sum(generators[supplier].probabilities[scenario] * generators[supplier].cost.calculate(JuMP.value(supply_real_time[supplier, scenario])) for scenario in scenarios) for supplier in suppliers)

    return supply_day_ahead_total, supply_real_time, supply_day_ahead_cost, supply_real_time_cost

end

function solve_consumption_problem(wind_powers::Vector{Float64}, market_prices::Vector{Float64}, loads::Vector{MarketClearing.Load})

    optimizer = optimizer_with_attributes(Ipopt.Optimizer, ("print_level" => 0))
    model = Model(optimizer)

    scenarios = 1:length(wind_powers)
    consumers = 1:length(loads)

    @variable(model, consumption_day_ahead[consumers])
    @variable(model, consumption_real_time[consumers, scenarios])

    @constraint(model, [consumer in consumers], 0 <= consumption_day_ahead[consumer] <= loads[consumer].capacity)
    @constraint(model, [consumer in consumers, scenario in scenarios], 0 <= consumption_real_time[consumer, scenario] <= loads[consumer].capacity)

    @objective(model, Max,
        sum(loads[consumer].utility.calculate(consumption_day_ahead[consumer]) for consumer in consumers) +
        sum(sum(loads[consumer].probabilities[scenario] * loads[consumer].utility.calculate(consumption_real_time[consumer, scenario]) for scenario in scenarios) for consumer in consumers) -
        sum(sum(market_prices[scenario] * (consumption_day_ahead[consumer] + consumption_real_time[consumer, scenario]) for scenario in scenarios) for consumer in consumers)
    )

    JuMP.optimize!(model)

    consumption_day_ahead_total = sum(JuMP.value(consumption_day_ahead[consumer]) for consumer in consumers)
    consumption_real_time = JuMP.value.(consumption_real_time)

    consumption_real_time_utility = sum(loads[consumer].utility.calculate(JuMP.value(consumption_day_ahead[consumer])) for consumer in consumers)
    consumption_real_time_utility = sum(sum(loads[consumer].probabilities[scenario] * loads[consumer].utility.calculate(JuMP.value(consumption_real_time[consumer, scenario])) for scenario in scenarios) for consumer in consumers)

    return consumption_day_ahead_total, consumption_real_time, consumption_real_time_utility, consumption_real_time_utility
end
