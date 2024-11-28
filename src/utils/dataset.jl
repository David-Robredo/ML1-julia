using DataFrames
using Statistics
using Printf
using Plots
using StatsPlots

function dataset_to_matrix(
    df::DataFrame, 
    target_column::Symbol, 
    input_type::Type=Float64
)
    @assert(target_column in propertynames(df), "Target column doesn't exist in DataFrame.")

    # Convert the input columns to a matrix of the specified type
    input_matrix = Matrix{input_type}(df[:, Not(target_column)])
    
    # Extract the target column as a vector. We don't convert to
    # any type as the output can be of arbtriary type.
    output_vector = df[:, target_column]
    
    return input_matrix, output_vector
end

function value_counts(df::DataFrame, target_variable::Symbol)
    counts = combine(groupby(df, target_variable), nrow => :Count)
    counts[!, :Percentage] = 100 .* counts[!, :Count] ./ sum(counts[!, :Count])
    return counts
end

function plot_value_counts(
    df::DataFrame, 
    target_variable::Symbol; 
    sort::Bool=true,
    kwargs...
)
    counts = value_counts(df, target_variable)
    if sort
        sort!(counts, :Count, rev=true)
    end

    p = bar(
        string.(counts[!, target_variable]), 
        counts[!, :Count],
        title = "Value Counts for $(string(target_variable))",
        ylabel = "Count",
        label = "Count",
        xticks = :auto,
        rotate = true,
        legend = false
    )

    # Disable scientific notation
    yformatter = x -> string(Int(round(x)))
    plot!(p, yformatter=yformatter)

    # Add percentage annotations below bars
    for i in 1:nrow(counts)
        annotate!(p, 
            i-0.45, 
            counts[i, :Count] - maximum(counts[!, :Count]) * 0.05,  # Offset above bar
            text("$(round(counts[i, :Percentage], digits=1))%", :black, :center)
        )
    end
    
    return p
end

function count_nulls(df::DataFrame)
    return DataFrame(
        Column = names(df), 
        Nulls = [sum(ismissing.(df[!, col])) for col in names(df)],
        Percentage = [100 * sum(ismissing.(df[!, col])) / nrow(df) for col in names(df)]
    )
end

function plot_null_counts(df::DataFrame; sort::Bool=true, kwargs...)
    null_counts = count_nulls(df)    
    if sort
        sort!(null_counts, :Nulls, rev=true)
    end

    p = bar(
        string.(null_counts[!, :Column]), 
        null_counts[!, :Nulls],
        title = "Null Counts by Column",
        ylabel = "Count",
        label = "Count",
        xticks = :auto,
        rotate = true,
        legend = false
    )

    # Disable scientific notation
    yformatter = x -> string(Int(round(x)))
    plot!(p, yformatter=yformatter)

    # Add percentage annotations below bars
    for i in 1:nrow(null_counts)
        annotate!(p, 
            i-0.45, 
            null_counts[i, :Nulls] - maximum(null_counts[!, :Nulls]) * 0.05,  # Offset above bar
            text("$(round(null_counts[i, :Percentage], digits=1))%", :black, :center)
        )
    end
    
    return p
end

function plot_heatmap(df::DataFrame, target_variable::Symbol)
    correlation_matrix = cor(Matrix(df[:, Not(target_variable)]))
    features = names(df[:, Not(target_variable)])

    heatmap(
        features, 
        features,
        correlation_matrix,
        color = :viridis,
        title = "Correlation Heatmap",
        aspect_ratio = :equal,
        xlabel = "Features",
        ylabel = "Features",
        size = (1000, 1000),
        colorbar_title = "Correlation",
        c = :coolwarm,
        xticks = (1:length(features), features),
        xrotation = 45,
        yticks = (1:length(features), features)
    )
end

function plot_boxplots(df::DataFrame; kwargs...)
    # Make boxplots only on numeric columns.
    numeric_cols = names(df, eltype.(eachcol(df)) .<: Number)
    
    if isempty(numeric_cols)
        error("No numeric columns found in the DataFrame.")
    end
    
    p = plot(
        layout=(1, length(numeric_cols)), 
        size=(200*length(numeric_cols), 400)
    )
    
    for (i, col) in enumerate(numeric_cols)
        boxplot!(
            df[!, col], 
            xlabel=string(col),
            subplot=i, 
            legend=false,
            kwargs...
        )
    end
    
    return p
end

@testset "dataset_to_matrix" begin
    # Dummy dataframe
    df = DataFrame(
        ID = 1:10,
        Age = rand(18:65, 10),
        Salary = rand(30000:100000, 10),
    )

    # Change dataframe to a 2D matrix of floats.
    inputs, outputs = dataset_to_matrix(df,  :Salary, Float64)

    @test typeof(inputs)  <: AbstractArray{Float64, 2}
    @test typeof(outputs) <: AbstractArray{Int64, 1}
end