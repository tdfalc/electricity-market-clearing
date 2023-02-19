using YAML

function get_settings(; overrides::Dict=Dict(), filename::String="./settings.yaml")
    settings = YAML.load_file(filename; dicttype=Dict{String,Any})
    nested_merge!(settings, overrides)
    return nested_parse(settings)
end

function nested_merge!(d::Dict...)
    return merge!(nested_merge!, d...)
end

nested_merge!(x::Any...) = x[end]

function nested_parse(d::Dict)
    return (; Dict(Symbol(k) => nested_parse(v) for (k, v) in d)...)
end

nested_parse(x::Any) = x
