using Flux;
using Test;

"""
Builds a Flux-based feedforward neural network for classification tasks.

# Arguments
- `numInputs::Int`: The number of input features.
- `topology::AbstractArray{<:Int,1}`: An array specifying the number of neurons 
    in each hidden layer.
- `numOutputs::Int`: The number of output classes.
- `transferFunctions::Union{AbstractArray{<:Function,1}, Function}`: An 
    array of activation functions for each hidden layer, or a single
    activation function for all layers. Defaults to sigmoid.

# Returns
- `Flux.Chain`: A Flux neural network model.
"""
function buildClassANN(
    numInputs::Int,
    topology::AbstractArray{<:Int,1},
    numOutputs::Int;
    transferFunctions::Union{AbstractArray{<:Function,1}, Function} = σ
)
    @assert(numInputs > 0, "The number of input attributes must be positive")
    @assert(numOutputs > 0, "The number of output attributes must be positive")
    
    # If transferFunctions is a single function, create an array with the function repeated
    if isa(transferFunctions, Function)
        transferFunctions = fill(transferFunctions, length(topology))
    end
    
    @assert(
        length(transferFunctions) == length(topology), 
        "There must be as many activation functions as hidden layers"
    )

    ann = Chain()
    numInputsLayer = numInputs

    # Build hidden layers
    for i in eachindex(topology)
        numNeurons = topology[i]
        activation = transferFunctions[i]
        ann = Chain(ann..., Dense(numInputsLayer, numNeurons, activation))
        numInputsLayer = numNeurons
    end

    # If using just a single output, apply sigmoid. Otherwise, apply softmax.
    if numOutputs == 1
        ann = Chain(ann..., Dense(numInputsLayer, 1, σ))
    else
        ann = Chain(ann..., Dense(numInputsLayer, numOutputs, identity))
        ann = Chain(ann..., softmax)
    end

    return ann
end;

@testset "buildClassANN" begin
    # Test with default transfer functions
    ann = buildClassANN(5, [10, 8], 3)
    @test ann isa Flux.Chain
    # 175 parameters (5*10+10 + 10*8+8 + 8*3+3)
    @test sum([length(layer) for layer in Flux.params(ann)]) == 175
    test_input = Array{Real,2}([0 1 2 3 4; 1 2 3 4 5])
    # Check shape of output
    @test size(ann(test_input')) == (3,2)
    # Check that outputs are constrained to sum 1.0 due to softmax
    @test all(sum(ann(test_input'), dims=1) .≈ 1.0)

    # Test with custom transfer functions
    ann = buildClassANN(7, [12, 10], 4; transferFunctions=relu)
    @test ann isa Flux.Chain

    # Test error handling
    @test_throws AssertionError buildClassANN(3, [6, 4], 1; transferFunctions=[tanh])
    @test_throws AssertionError buildClassANN(-1, [6, 4], 1)
    @test_throws AssertionError buildClassANN(3, [6, 4], -1)
end