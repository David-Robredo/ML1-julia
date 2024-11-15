using Random;

"""
Generate fold assignments for N items into k folds.

# Arguments
- `N::Int64`: Number of items to distribute across folds.
- `k::Int64`: Number of folds to create.

# Returns
- `Vector{Int64}`: A shuffled vector of length N containing 
    fold assignments (1 to k)
"""
function crossvalidation(N::Int64, k::Int64)
    @assert(N > 0, "Number of samples (N) must be greater than 0.")
    @assert(k > 0, "Number of folds (k) must be greater than 0.")

    base_vector = 1:k

    repeat_vector = repeat(base_vector, ceil(Int, N / k))
    subset_vector = repeat_vector[1:N]
    shuffle!(subset_vector)

    return subset_vector
end

"""
Generate stratified fold assignments for classification problems.
Maintains class proportions across folds.

# Arguments
- `targets::AbstractArray{Bool,2}`: Target matrix where each column 
    represents a class
- `k::Int64`: Number of folds to create

# Returns
- `Vector{Int64}`: A vector containing fold assignments (1 to k) preserving 
    class distributions
"""
function crossvalidation(targets::AbstractArray{Bool,2}, k::Int64)
    npatterns,nclasses = size(targets)
    
    indices = Array{Int64,1}(undef, npatterns);
    [indices[targets[:,n]] .= crossvalidation(sum(targets[:,n]), k) for n in 1:nclasses]
    
    return indices
end

"""
Generate stratified fold assignments for classification problems.
Maintains class proportions across folds.

# Arguments
- `targets::AbstractArray{<:Any,1}`: Vector of class labels
- `k::Int64`: Number of folds to create

# Returns
- `Vector{Int64}`: A vector containing fold assignments (1 to k) preserving 
    class distributions
"""
function crossvalidation(targets::AbstractArray{<:Any,1}, k::Int64)
    classes = [targets.==(cls) for cls in unique(targets)]
    npatterns = size(targets)
    
    indices = Array{Int64,1}(undef, npatterns);
    [indices[cls_map] .= crossvalidation(sum(cls_map), k) for cls_map in classes]
    return indices
end

@testset "crossvalidation(N, k)" begin
    # Test basic folds.
    Random.seed!(42)
    result = crossvalidation(10, 3)
    @test length(result) == 10
    @test sort(unique(result)) == collect(1:3)

    # Test with invalid input
    @test_throws AssertionError crossvalidation(0, 3)
    @test_throws AssertionError crossvalidation(10, 0)
end

@testset "crossvalidation(matrix, k)" begin
    Random.seed!(42)
    targets = Bool[1 0; 1 0; 0 1; 0 1]
    result = crossvalidation(targets, 2)
    @test length(result) == 4
    @test sort(unique(result)) == collect(1:2)
    
    # Check class balance.
    # Both classes should appear in both folds
    class1_indices = targets[:, 1]
    class1_folds = result[class1_indices]
    @test length(unique(class1_folds)) == 2
end

@testset "crossvalidation(data, k)" begin
    Random.seed!(42)
    targets = ["A", "A", "B", "C", "C"]
    result = crossvalidation(targets, 2)
    @test length(result) == 5
    @test sort(unique(result)) == collect(1:2)
    
    # Check class balance.
    # Both "A" and "C" should appear in both folds
    classA_indices = targets .== "A"
    classA_folds = result[classA_indices]
    @test length(unique(classA_folds)) == 2

    classC_indices = targets .== "C"
    classC_folds = result[classC_indices]
    @test length(unique(classC_folds)) == 2
end