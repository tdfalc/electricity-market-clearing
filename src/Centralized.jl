function solve_centralized_model(wind_powers::Vector{Float64}, loads::Vector{MarketClearing.Load}, generators::Vector{MarketClearing.Generator})

    optimizer = optimizer_with_attributes(Ipopt.Optimizer, ("print_level" => 0))
    model = Model(optimizer)

    scenarios = 1:length(wind_powers)
    suppliers = 1:length(generators)
    consumers = 1:length(loads)

    @variable(model, supply_day_ahead[suppliers])
    @variable(model, consumption_day_ahead[consumers])
    @variable(model, supply_real_time[suppliers, scenarios])
    @variable(model, consumption_real_time[consumers, scenarios])

    @constraint(model, [supplier in suppliers], 0 <= supply_day_ahead[supplier] <= generators[supplier].capacity)
    @constraint(model, [supplier in suppliers, scenario in scenarios], -generators[supplier].capacity <= supply_real_time[supplier, scenario] <= generators[supplier].capacity)

    @constraint(model, [consumer in consumers], 0 <= consumption_day_ahead[consumer] <= loads[consumer].capacity)
    @constraint(model, [consumer in consumers, scenario in scenarios], 0 <= consumption_real_time[consumer, scenario] <= loads[consumer].capacity)

    @constraint(model, power_balance[scenario in scenarios],
        wind_powers[scenario] +
        sum(supply_day_ahead[supplier] for supplier in suppliers) +
        sum(supply_real_time[supplier, scenario] for supplier in suppliers) -
        sum(consumption_day_ahead[consumer] for consumer in consumers) -
        sum(consumption_real_time[consumer, scenario] for consumer in consumers) == 0,
    )

    @objective(model, Max,
        sum(loads[consumer].utility.calculate(consumption_day_ahead[consumer]) for consumer in consumers) -
        sum(generators[supplier].cost.calculate(supply_day_ahead[supplier]) for supplier in suppliers) +
        sum(sum(loads[consumer].probabilities[scenario] * loads[consumer].utility.calculate(consumption_real_time[consumer, scenario]) for scenario in scenarios) for consumer in consumers) -
        sum(sum(generators[supplier].probabilities[scenario] * generators[supplier].cost.calculate(supply_real_time[supplier, scenario]) for scenario in scenarios) for supplier in suppliers),
    )

    JuMP.optimize!(model)
    if JuMP.termination_status(model) != MOI.LOCALLY_SOLVED
        @info "Centralized: Optimization Failed. Status: $(JuMP.termination_status(model))"
        return
    end

    supply_day_ahead_total = sum(JuMP.value(supply_day_ahead[supplier]) for supplier in suppliers)
    consumption_day_ahead_total = sum(JuMP.value(consumption_day_ahead[consumer]) for consumer in consumers)

    social_welfare = JuMP.objective_value(model)
    clearing_price = sum(JuMP.dual(power_balance[scenario]) for scenario in scenarios)

    return social_welfare, supply_day_ahead_total, consumption_day_ahead_total, clearing_price
end
