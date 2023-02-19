function solve_mixed_complementarity_model(wind_powers::Vector{Float64}, loads::Vector{Load}, generators::Vector{Generator}; M::Int64=10000)

    optimizer = optimizer_with_attributes(SCIP.Optimizer, ("display/verblevel" => 0))
    model = Model(optimizer)

    scenarios = 1:length(wind_powers)
    suppliers = 1:length(generators)
    consumers = 1:length(loads)

    @variables model begin
        supply_day_ahead[suppliers] # Day ahead supply primals.
        consumption_day_ahead[consumers] # Day ahead consumption primals.

        supply_real_time[suppliers, scenarios] # Real time supply primals.
        consumption_real_time[consumers, scenarios] # Real time consumption primals.

        equilibrium_price[scenarios] >= 0 # Current equilibrium/clearing price.

        supply_day_ahead_dual[1:2, suppliers] >= 0 # Day ahead supply bound duals.
        supply_real_time_dual[1:2, suppliers, scenarios] >= 0 # Real time supply bound duals.

        consumption_day_ahead_dual[1:2, consumers] >= 0 # Day ahead consumption bound duals.
        consumption_real_time_dual[1:2, consumers, scenarios] >= 0 # Real time consumption bound duals.

        supply_day_ahead_binary[1:2, supplier in suppliers], Bin # Day ahead supply bound complementarity binaries.
        supply_real_time_binary[1:2, supplier in suppliers, scenario in scenarios], Bin # Real time supply bound complementarity binaries.

        consumption_day_ahead_binary[1:2, consumer in consumers], Bin # Day ahead consumption bound complementarity binaries.
        consumption_real_time_binary[1:2, consumer in consumers, scenario in scenarios], Bin # Real time consumption bound complementarity binaries.
    end

    # Price setter problem - sets equilibrium price based on day ahead and real time supply and consumption.
    @constraint(model, [scenario in scenarios],
        0 == wind_powers[scenario] +
             sum(supply_day_ahead[supplier] for supplier in suppliers) +
             sum(supply_real_time[supplier, scenario] for supplier in suppliers) -
             sum(consumption_day_ahead[consumer] for consumer in consumers) -
             sum(consumption_real_time[consumer, scenario] for consumer in consumers),
    )

    # Supplier problem - optimizes generation based on beliefs about market price.
    @constraint(model, [supplier in suppliers],
        0 == generators[supplier].cost.alpha * supply_day_ahead[supplier] -
             sum(equilibrium_price[scenario] for scenario in scenarios) -
             supply_day_ahead_dual[1, supplier] +
             supply_day_ahead_dual[2, supplier]
    )

    @constraint(model, [supplier in suppliers], 0 <= supply_day_ahead[supplier])
    @constraint(model, [supplier in suppliers], supply_day_ahead[supplier] <= M * supply_day_ahead_binary[1, supplier])
    @constraint(model, [supplier in suppliers], supply_day_ahead_dual[1, supplier] <= M * (1 - supply_day_ahead_binary[1, supplier]))

    @constraint(model, [supplier in suppliers], 0 <= -supply_day_ahead[supplier] + generators[supplier].capacity)
    @constraint(model, [supplier in suppliers], -supply_day_ahead[supplier] + generators[supplier].capacity <= M * supply_day_ahead_binary[2, supplier])
    @constraint(model, [supplier in suppliers], supply_day_ahead_dual[2, supplier] <= M * (1 - supply_day_ahead_binary[2, supplier]))

    @constraint(model, [supplier in suppliers, scenario in scenarios],
        0 == generators[supplier].probabilities[scenario] * generators[supplier].cost.alpha * supply_real_time[supplier, scenario] -
             equilibrium_price[scenario] -
             supply_real_time_dual[1, supplier, scenario] + supply_real_time_dual[2, supplier, scenario]
    )

    @constraint(model, [supplier in suppliers, scenario in scenarios], 0 <= supply_real_time[supplier, scenario] + generators[supplier].capacity)
    @constraint(model, [supplier in suppliers, scenario in scenarios], supply_real_time[supplier, scenario] + generators[supplier].capacity <= M * supply_real_time_binary[1, supplier, scenario])
    @constraint(model, [supplier in suppliers, scenario in scenarios], supply_real_time_dual[1, supplier, scenario] <= M * (1 - supply_real_time_binary[1, supplier, scenario]))

    @constraint(model, [supplier in suppliers, scenario in scenarios], 0 <= -supply_real_time[supplier, scenario] + generators[supplier].capacity)
    @constraint(model, [supplier in suppliers, scenario in scenarios], -supply_real_time[supplier, scenario] + generators[supplier].capacity <= M * supply_real_time_binary[2, supplier, scenario])
    @constraint(model, [supplier in suppliers, scenario in scenarios], supply_real_time_dual[2, supplier, scenario] <= M * (1 - supply_real_time_binary[2, supplier, scenario]))

    # Consumer problem - optimizes load based on beliefs about market price.
    @constraint(model, [consumer in consumers],
        0 == sum(equilibrium_price[scenario] for scenario in scenarios) -
             (loads[consumer].utility.tau - loads[consumer].utility.beta * consumption_day_ahead[consumer]) - consumption_day_ahead_dual[1, consumer] + consumption_day_ahead_dual[2, consumer]
    )

    @constraint(model, [consumer in consumers], 0 <= consumption_day_ahead[consumer])
    @constraint(model, [consumer in consumers], consumption_day_ahead[consumer] <= M * consumption_day_ahead_binary[1, consumer])
    @constraint(model, [consumer in consumers], consumption_day_ahead_dual[1, consumer] <= M * (1 - consumption_day_ahead_binary[1, consumer]))

    @constraint(model, [consumer in consumers], 0 <= -consumption_day_ahead[consumer] + loads[consumer].capacity)
    @constraint(model, [consumer in consumers], -consumption_day_ahead[consumer] + loads[consumer].capacity <= M * consumption_day_ahead_binary[2, consumer])
    @constraint(model, [consumer in consumers], consumption_day_ahead_dual[2, consumer] <= M * (1 - consumption_day_ahead_binary[2, consumer]))

    @constraint(model, [consumer in consumers, scenario in scenarios],
        0 == equilibrium_price[scenario] -
             loads[consumer].probabilities[scenario] * (loads[consumer].utility.tau - loads[consumer].utility.beta * consumption_real_time[consumer, scenario]) -
             consumption_real_time_dual[1, consumer, scenario] + consumption_real_time_dual[2, consumer, scenario]
    )

    @constraint(model, [consumer in consumers, scenario in scenarios], 0 <= consumption_real_time[consumer, scenario])
    @constraint(model, [consumer in consumers, scenario in scenarios], consumption_real_time[consumer, scenario] <= M * consumption_real_time_binary[1, consumer, scenario])
    @constraint(model, [consumer in consumers, scenario in scenarios], consumption_real_time_dual[1, consumer, scenario] <= M * (1 - consumption_real_time_binary[1, consumer, scenario]))

    @constraint(model, [consumer in consumers, scenario in scenarios], 0 <= -consumption_real_time[consumer, scenario] + loads[consumer].capacity)
    @constraint(model, [consumer in consumers, scenario in scenarios], -consumption_real_time[consumer, scenario] + loads[consumer].capacity <= M * consumption_real_time_binary[2, consumer, scenario])
    @constraint(model, [consumer in consumers, scenario in scenarios], consumption_real_time_dual[2, consumer, scenario] <= M * (1 - consumption_real_time_binary[2, consumer, scenario]))

    JuMP.optimize!(model)
    if JuMP.termination_status(model) != OPTIMAL
        @info "Mixed Complementarity: Optimization Failed. Status: $(JuMP.termination_status(model))"
        return
    end

    supply_day_ahead_cost = sum(generators[supplier].cost.calculate(JuMP.value(supply_day_ahead[supplier])) for supplier in suppliers)
    supply_real_time_cost = sum(sum(generators[supplier].probabilities[scenario] * generators[supplier].cost.calculate(JuMP.value(supply_real_time[supplier, scenario])) for scenario in scenarios) for supplier in suppliers)

    consumption_day_ahead_utility = sum(loads[consumer].utility.calculate(JuMP.value(consumption_day_ahead[consumer])) for consumer in consumers)
    consumption_real_time_utility = sum(sum(loads[consumer].probabilities[scenario] * loads[consumer].utility.calculate(JuMP.value(consumption_real_time[consumer, scenario])) for scenario in scenarios) for consumer in consumers)

    supply_day_ahead_total = sum(JuMP.value(supply_day_ahead[supplier]) for supplier in suppliers)
    consumption_day_ahead_total = sum(JuMP.value(consumption_day_ahead[consumer]) for consumer in consumers)

    social_welfare = consumption_day_ahead_utility + consumption_real_time_utility - supply_day_ahead_cost - supply_real_time_cost
    equilibrium_price = sum(JuMP.value(equilibrium_price[scenario]) for scenario in scenarios)

    return social_welfare, supply_day_ahead_total, consumption_day_ahead_total, equilibrium_price
end
