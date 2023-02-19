struct Utility
    tau::Float64
    beta::Float64
    calculate::Function
    function Utility(tau, beta)
        new(tau, beta, (x) -> tau * x - 0.5 * beta * x^2)
    end
end

struct Cost
    alpha::Float64
    calculate::Function
    function Cost(alpha)
        new(alpha, (x) -> 0.5 * alpha * x^2)
    end
end

struct Load
    capacity::Float64
    probabilities::Any
    utility::Utility
end

struct Generator
    capacity::Float64
    probabilities::Any
    cost::Cost
end
