function generate_samples(mean::Float64, variance::Float64, num_samples::Int, seed::Int; sorted=false)
    #Random.seed!(seed)
    #distribution = Normal(mean, variance^0.5)
    # samples = rand(distribution, num_samples)
    # if sorted
    #     return sort(samples)
    # return samples


    # weighted_values = zeros(100)
    # weighted_values_sq = zeros(100)
    # global weights = diff(vcat(0, tweaked_cdf))
    # for weight in 1:length(weights)
    #     weighted_values[weight] = weights[weight] * samples[weight]
    #     weighted_values_sq[weight] = weights[weight] * samples[weight] * samples[weight]
    # end


    wind_scenarios_table = CSV.read("./wind_scenarios.csv", DataFrame)
    return sort(wind_scenarios_table[:, :L])
end

function calculate_reference_pdf(samples::Vector{Float64})
    reference_probability = 1 / length(samples)
    return fill(reference_probability, length(samples))
end

function calculate_tweaked_pdf(reference_pdf::Vector{Float64}, gamma::Float64, delta::Float64)
    reference_cdf = cumsum(reference_pdf)
    # Using norm avoids floating point error
    tweaked_cdf = [(delta * prob^gamma) / (delta * prob^gamma + norm(1 - prob)^gamma) for prob in reference_cdf]
    return diff(vcat(0, tweaked_cdf))
end

function print_tweaked_scenario_metrics(wind_powers::Vector{Float64}, tweaked_pdf::Vector{Float64})
    mean = sum(wind_powers .* tweaked_pdf)
    variance = sum((wind_powers .^ 2) .* tweaked_pdf)
    println("Mean: $(mean); Variance: $(variance)")
end

"Convenience function to generate wind power samples and produce reference and tweaked PDFs."
function build_data(mean::Float64, variance::Float64, num_samples::Int, seed::Int, gamma::Float64, delta::Float64)
    wind_powers = generate_samples(mean, variance, num_samples, seed; sorted=true)
    reference_pdf = calculate_reference_pdf(wind_powers)
    tweaked_pdf = calculate_tweaked_pdf(reference_pdf, gamma, delta)
    print_tweaked_scenario_metrics(wind_powers, tweaked_pdf)
    return wind_powers, reference_pdf, tweaked_pdf
end

