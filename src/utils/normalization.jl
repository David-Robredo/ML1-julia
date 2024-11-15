using Statistics

"""
Normalize a 2D array along each column dimension using provided mean and 
standard deviation vectors.

# Arguments
- `data::AbstractArray{<:Real,2}`: Input 2D array of real numbers
- `mean::AbstractVector{<:Real}`: Vector of means for each column
- `std::AbstractVector{<:Real}`: Vector of standard deviations for each column

# Returns
- `AbstractArray{T,2}`: Normalized array with same dimensions as input
"""
function normalize(
    data::AbstractArray{<:Real,2}, 
    μ::AbstractVector{<:Real}, 
    σ::AbstractVector{<:Real}
)
    # Replace std with value of 0 to avoid division by 0.
    σ = map(s -> s == 0.0 ? 1.0 : s, σ)
    return (data .- reshape(μ,1,:)) ./ reshape(σ,1,:)
end

"""
Compute mean and standard deviation of a 2D array along all samples for each 
of the columns.

# Arguments
- `data::AbstractArray{<:Real,2}`: Input 2D array of real numbers

# Returns
- `Tuple{Vector{<:Real}, Vector{<:Real}}`: Tuple containing:
  - Vector of means for each column
  - Vector of standard deviations for each column
"""
compute_μσ(data::AbstractArray{<:Real,2}) = mean(data, dims=1)[1,:], std(data, dims=1)[1,:]

@testset "Normalization" begin
    values = [
        3 6 1;
        2 5 1;
        1 4 1;
    ]
    μ, σ = compute_μσ(values)
    @test μ ≈ [2.0, 5.0, 1.0]
    @test σ ≈ [1.0, 1.0, 0.0]

    normalized = normalize(values, μ, σ)
    μ, σ = compute_μσ(normalized)
    @test μ ≈ [0.0, 0.0, 0.0]
    @test σ ≈ [1.0, 1.0, 0.0]
end